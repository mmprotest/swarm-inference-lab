using System.Text.Json;

namespace SwarmBootstrap;

internal sealed class RuntimeInstaller
{
    private readonly PathInfo _layout;
    private readonly string _payloadDirectory;
    private readonly IBoundedProcessRunner _processes;
    private readonly BackendDetector _backendDetector;
    private readonly ServiceLifecycle _services;
    private readonly BootstrapLogger _logger;
    private readonly TimeSpan _processTimeout;
    private readonly bool _keepFailedStaging;

    public RuntimeInstaller(
        PathInfo layout,
        string payloadDirectory,
        IBoundedProcessRunner processes,
        BootstrapLogger logger,
        TimeSpan processTimeout,
        bool keepFailedStaging)
    {
        _layout = layout;
        _payloadDirectory = Path.GetFullPath(payloadDirectory);
        _processes = processes;
        _logger = logger;
        _processTimeout = processTimeout;
        _keepFailedStaging = keepFailedStaging;
        _backendDetector = new BackendDetector(processes, logger);
        _services = new ServiceLifecycle(processes, logger, TimeSpan.FromSeconds(90));
    }

    public async Task<InstallationResult> ExecuteAsync(
        string requestedOperation,
        BackendProfile requestedBackend,
        bool allowDowngrade,
        CancellationToken cancellationToken)
    {
        BackendDetector.ValidatePlatform();
        ReleaseManifest manifest = HashVerifier.LoadAndVerifyPayload(_payloadDirectory);
        _layout.EnsureBaseDirectories();
        InstallRecord? previous = InstallRecord.Load(_layout);
        string operation = ResolveOperation(requestedOperation, previous, manifest, allowDowngrade);
        ServiceSnapshot serviceSnapshot = await _services.CaptureAsync(cancellationToken)
            .ConfigureAwait(false);
        if (serviceSnapshot.HadService)
        {
            await _services.StopAsync(serviceSnapshot, cancellationToken).ConfigureAwait(false);
        }

        using RuntimeTransaction transaction = new(_layout, _logger, _keepFailedStaging);
        bool pathAdded = false;
        try
        {
            transaction.MoveActiveAside(previous);
            string controlledUv = CacheVerifiedUv(manifest.Uv);
            BackendDetectionResult detection = requestedBackend == BackendProfile.Auto
                ? await _backendDetector.DetectAsync(cancellationToken).ConfigureAwait(false)
                : new BackendDetectionResult(
                    requestedBackend,
                    requestedBackend == BackendProfile.Cuda,
                    $"explicit {requestedBackend.ToString().ToLowerInvariant()} override",
                    null,
                    null);
            List<BackendProfile> candidates = CandidateOrder(requestedBackend, detection);
            CandidateRuntime? selected = null;
            string? rejectedProfile = null;
            string? rejectionReason = null;
            foreach (BackendProfile candidate in candidates)
            {
                string staging = transaction.CreateStagingRuntime();
                try
                {
                    selected = await InstallCandidateAsync(
                            manifest,
                            controlledUv,
                            candidate,
                            staging,
                            cancellationToken)
                        .ConfigureAwait(false);
                    break;
                }
                catch (Exception exception) when (
                    requestedBackend == BackendProfile.Auto
                    && candidate == BackendProfile.Cuda
                    && exception is not OperationCanceledException
                    && exception is not UnauthorizedAccessException)
                {
                    rejectedProfile = "cuda";
                    rejectionReason = BootstrapLogger.Redact(exception.Message);
                    _logger.Error(
                        $"automatic CUDA candidate rejected; CPU fallback will be installed: {rejectionReason}");
                    transaction.DiscardStaging();
                }
            }

            if (selected is null)
            {
                throw new DependencyInstallException("no runtime profile produced a valid candidate");
            }

            transaction.PublishStaging();
            selected = await RebindPublishedRuntimeAsync(
                    manifest,
                    controlledUv,
                    selected,
                    cancellationToken)
                .ConfigureAwait(false);
            CacheVerifiedPayload(manifest);
            UserPathRegistration.Add(_layout.ScriptsPath);
            pathAdded = true;
            await _services.RestoreAsync(
                    _layout.SwarmExecutable,
                    _layout.StateRoot,
                    cancellationToken)
                .ConfigureAwait(false);
            InstallRecord record = BuildInstallRecord(
                operation,
                manifest,
                detection.Candidate,
                selected,
                rejectedProfile,
                rejectionReason,
                previous);
            record.SaveAtomic(_layout);
            transaction.Commit();
            return new InstallationResult(
                operation,
                manifest.Version,
                manifest.GitTag,
                selected.Backend.ToString().ToLowerInvariant(),
                rejectedProfile,
                rejectionReason,
                _layout.Root,
                _layout.StateRoot,
                _logger.LogPath,
                selected.Doctor);
        }
        catch (Exception installationFailure)
        {
            Exception? serviceCleanupFailure = null;
            try
            {
                ServiceSnapshot current = await _services.CaptureAsync(cancellationToken)
                    .ConfigureAwait(false);
                await _services.StopAsync(current, cancellationToken).ConfigureAwait(false);
                if (!serviceSnapshot.HadService && current.HadService)
                {
                    await _services.RemoveAsync(current, cancellationToken).ConfigureAwait(false);
                }
            }
            catch (Exception exception)
            {
                serviceCleanupFailure = exception;
            }

            transaction.Rollback();
            if (previous is null && pathAdded)
            {
                UserPathRegistration.Remove(_layout.ScriptsPath);
            }

            if (previous is not null)
            {
                try
                {
                    await _services.RestoreAsync(
                            _layout.SwarmExecutable,
                            _layout.StateRoot,
                            cancellationToken)
                        .ConfigureAwait(false);
                }
                catch (Exception rollbackServiceFailure)
                {
                    throw new ServiceLifecycleException(
                        $"upgrade failed ({installationFailure.Message}); runtime rollback succeeded, "
                        + $"but previous service restoration failed ({rollbackServiceFailure.Message})",
                        rollbackServiceFailure);
                }

                string cleanup = serviceCleanupFailure is null
                    ? string.Empty
                    : $"; service cleanup warning: {serviceCleanupFailure.Message}";
                throw new UpgradeRollbackException(
                    $"upgrade failed and version {previous.ProductVersion} was restored: "
                    + $"{installationFailure.Message}{cleanup}",
                    installationFailure);
            }

            if (serviceCleanupFailure is not null)
            {
                throw new ServiceLifecycleException(
                    $"installation failed ({installationFailure.Message}); "
                    + $"service cleanup also failed ({serviceCleanupFailure.Message})",
                    serviceCleanupFailure);
            }

            throw;
        }
    }

    public async Task<JsonElement> DoctorAsync(CancellationToken cancellationToken)
    {
        BackendDetector.ValidatePlatform();
        InstallRecord record = InstallRecord.Load(_layout)
            ?? throw new DoctorFailureException("native installation record is missing");
        ReleaseManifest manifest = HashVerifier.LoadAndVerifyPayload(_layout.PayloadCache);
        if (manifest.Version != record.ProductVersion || !File.Exists(_layout.SwarmExecutable))
        {
            throw new DoctorFailureException("installed runtime and installation record do not agree");
        }

        ProcessResult result = await _processes.RunAsync(
            new ProcessRequest(
                _layout.SwarmExecutable,
                ["node", "doctor", "--json"],
                TimeSpan.FromSeconds(Math.Min(180, _processTimeout.TotalSeconds))),
            cancellationToken).ConfigureAwait(false);
        return ParseDoctor(result, record.SelectedBackend == "cuda" ? BackendProfile.Cuda : BackendProfile.Cpu);
    }

    public async Task UninstallAsync(bool purgeState, CancellationToken cancellationToken)
    {
        BackendDetector.ValidatePlatform();
        ServiceSnapshot snapshot = await _services.CaptureAsync(cancellationToken).ConfigureAwait(false);
        await _services.StopAsync(snapshot, cancellationToken).ConfigureAwait(false);
        await _services.RemoveAsync(snapshot, cancellationToken).ConfigureAwait(false);
        UserPathRegistration.Remove(_layout.ScriptsPath);
        foreach (string directory in new[]
                 {
                     _layout.Runtime,
                     _layout.PayloadCache,
                     _layout.App,
                     _layout.Previous,
                 })
        {
            DeleteOwnedDirectory(directory, _layout.Root);
        }

        if (purgeState)
        {
            string localAppData = Environment.GetEnvironmentVariable("LOCALAPPDATA")
                ?? Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            string expected = Path.Combine(Path.GetFullPath(localAppData), "SwarmInference");
            if (!string.Equals(
                    Path.TrimEndingDirectorySeparator(expected),
                    Path.TrimEndingDirectorySeparator(_layout.StateRoot),
                    StringComparison.OrdinalIgnoreCase))
            {
                throw new PermissionFailureException("durable state path is not the owned product state root");
            }

            if (Directory.Exists(_layout.StateRoot))
            {
                Directory.Delete(_layout.StateRoot, recursive: true);
            }

            _logger.Info("durable cluster state was explicitly purged");
        }
        else
        {
            _logger.Info($"durable cluster state preserved at {_layout.StateRoot}");
        }
    }

    private static string ResolveOperation(
        string requested,
        InstallRecord? previous,
        ReleaseManifest manifest,
        bool allowDowngrade)
    {
        if (requested == "install" && previous is not null)
        {
            requested = VersionPolicy.Compare(manifest.Version, previous.ProductVersion) == 0
                ? "repair"
                : "upgrade";
        }

        if (requested == "upgrade" && previous is null)
        {
            throw new ManifestException("upgrade requires an existing native installation record");
        }

        if (requested == "repair"
            && previous is not null
            && VersionPolicy.Compare(manifest.Version, previous.ProductVersion) != 0)
        {
            throw new ManifestException("repair payload version must match the installed version");
        }

        if (previous is not null
            && VersionPolicy.Compare(manifest.Version, previous.ProductVersion) < 0
            && !allowDowngrade)
        {
            throw new ManifestException(
                $"downgrade from {previous.ProductVersion} to {manifest.Version} is refused; "
                + "use --allow-downgrade only for documented recovery");
        }

        return requested;
    }

    internal static List<BackendProfile> CandidateOrder(
        BackendProfile requested,
        BackendDetectionResult detection)
    {
        if (requested == BackendProfile.Cpu)
        {
            return [BackendProfile.Cpu];
        }

        if (requested == BackendProfile.Cuda)
        {
            return [BackendProfile.Cuda];
        }

        return detection.Candidate == BackendProfile.Cuda
            ? [BackendProfile.Cuda, BackendProfile.Cpu]
            : [BackendProfile.Cpu];
    }

    private string CacheVerifiedUv(UvAsset uvAsset)
    {
        string tools = Path.Combine(_layout.PayloadCache, "tools", uvAsset.Version);
        Directory.CreateDirectory(tools);
        string destination = Path.Combine(tools, "uv.exe");
        CopyVerified(Path.Combine(_payloadDirectory, uvAsset.Filename), destination, uvAsset);
        return destination;
    }

    private async Task<CandidateRuntime> InstallCandidateAsync(
        ReleaseManifest manifest,
        string uvExecutable,
        BackendProfile backend,
        string staging,
        CancellationToken cancellationToken)
    {
        FileAsset profile = backend == BackendProfile.Cuda
            ? manifest.RuntimeProfiles.Cuda
            : manifest.RuntimeProfiles.Cpu;
        string profilePath = Path.Combine(_payloadDirectory, profile.Filename);
        string wheelPath = Path.Combine(_payloadDirectory, manifest.Wheel.Filename);
        Dictionary<string, string> environment = new(StringComparer.OrdinalIgnoreCase)
        {
            ["UV_CACHE_DIR"] = Path.Combine(_layout.PayloadCache, "uv-cache"),
            ["UV_PYTHON_INSTALL_DIR"] = Path.Combine(_layout.PayloadCache, "python"),
            ["UV_MANAGED_PYTHON"] = "1",
            ["UV_NO_SYSTEM_CONFIG"] = "1",
            ["UV_NO_PROGRESS"] = "1",
            ["PYTHONUTF8"] = "1",
            ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1",
        };
        await RunCheckedAsync(
            uvExecutable,
            [
                "--no-config",
                "python",
                "install",
                "--no-bin",
                "--no-registry",
                manifest.Python.Version,
            ],
            environment,
            "managed Python installation",
            cancellationToken).ConfigureAwait(false);
        await RunCheckedAsync(
            uvExecutable,
            ["--no-config", "venv", "--python", manifest.Python.Version, "--managed-python", staging],
            environment,
            "managed environment creation",
            cancellationToken).ConfigureAwait(false);
        string python = Path.Combine(staging, "Scripts", "python.exe");
        await RunCheckedAsync(
            uvExecutable,
            [
                "--no-config",
                "pip",
                "sync",
                "--python",
                python,
                // Exact versions plus --require-hashes prevent dependency drift while
                // allowing the pinned +cpu/+cu130 torch build on its dedicated index.
                "--index-strategy",
                "unsafe-best-match",
                "--require-hashes",
                profilePath,
            ],
            environment,
            $"{backend.ToString().ToLowerInvariant()} dependency synchronization",
            cancellationToken).ConfigureAwait(false);
        await RunCheckedAsync(
            uvExecutable,
            [
                "--no-config",
                "pip",
                "install",
                "--python",
                python,
                "--no-deps",
                "--reinstall",
                wheelPath,
            ],
            environment,
            "embedded application wheel installation",
            cancellationToken).ConfigureAwait(false);
        return await ValidateRuntimeAsync(
                manifest,
                backend,
                staging,
                profile,
                environment,
                cancellationToken)
            .ConfigureAwait(false);
    }

    private async Task<CandidateRuntime> RebindPublishedRuntimeAsync(
        ReleaseManifest manifest,
        string uvExecutable,
        CandidateRuntime candidate,
        CancellationToken cancellationToken)
    {
        Dictionary<string, string> environment = RuntimeEnvironment();
        await RunCheckedAsync(
            uvExecutable,
            [
                "--no-config",
                "pip",
                "install",
                "--python",
                _layout.PythonExecutable,
                "--no-deps",
                "--reinstall",
                Path.Combine(_payloadDirectory, manifest.Wheel.Filename),
            ],
            environment,
            "stable entry-point rebinding",
            cancellationToken).ConfigureAwait(false);
        FileAsset profile = candidate.Backend == BackendProfile.Cuda
            ? manifest.RuntimeProfiles.Cuda
            : manifest.RuntimeProfiles.Cpu;
        return await ValidateRuntimeAsync(
                manifest,
                candidate.Backend,
                _layout.Runtime,
                profile,
                environment,
                cancellationToken)
            .ConfigureAwait(false);
    }

    private async Task<CandidateRuntime> ValidateRuntimeAsync(
        ReleaseManifest manifest,
        BackendProfile backend,
        string runtime,
        FileAsset profile,
        IReadOnlyDictionary<string, string> environment,
        CancellationToken cancellationToken)
    {
        string python = Path.Combine(runtime, "Scripts", "python.exe");
        await RunCheckedAsync(
            python,
            [
                "-I",
                "-c",
                "import json, pathlib, swarm_inference, swarm_inference.cli; "
                + "print(json.dumps({'version': swarm_inference.__version__, "
                + "'module': str(pathlib.Path(swarm_inference.__file__).resolve())}))",
            ],
            environment,
            "installed import validation",
            cancellationToken).ConfigureAwait(false);
        string swarm = Path.Combine(runtime, "Scripts", "swarm.exe");
        ProcessResult version = await RunCheckedAsync(
            swarm,
            ["--version"],
            environment,
            "installed version validation",
            cancellationToken).ConfigureAwait(false);
        if (!string.Equals(version.StandardOutput.Trim(), manifest.Version, StringComparison.Ordinal))
        {
            throw new DoctorFailureException(
                $"installed version '{version.StandardOutput.Trim()}' does not match {manifest.Version}");
        }

        ProcessResult doctor = await _processes.RunAsync(
            new ProcessRequest(
                swarm,
                ["node", "doctor", "--json"],
                TimeSpan.FromSeconds(Math.Min(180, _processTimeout.TotalSeconds)),
                Environment: environment),
            cancellationToken).ConfigureAwait(false);
        JsonElement doctorDocument = ParseDoctor(doctor, backend);
        string selected = doctorDocument.GetProperty("backend_selection")
            .GetProperty("selected_backend")
            .GetString() ?? string.Empty;
        return new CandidateRuntime(
            backend,
            runtime,
            doctorDocument,
            selected,
            profile.Filename,
            profile.Sha256);
    }

    private Dictionary<string, string> RuntimeEnvironment()
    {
        return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["UV_CACHE_DIR"] = Path.Combine(_layout.PayloadCache, "uv-cache"),
            ["UV_PYTHON_INSTALL_DIR"] = Path.Combine(_layout.PayloadCache, "python"),
            ["UV_MANAGED_PYTHON"] = "1",
            ["UV_NO_SYSTEM_CONFIG"] = "1",
            ["UV_NO_PROGRESS"] = "1",
            ["PYTHONUTF8"] = "1",
            ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1",
        };
    }

    private async Task<ProcessResult> RunCheckedAsync(
        string executable,
        IReadOnlyList<string> arguments,
        IReadOnlyDictionary<string, string> environment,
        string stage,
        CancellationToken cancellationToken)
    {
        ProcessResult result = await _processes.RunAsync(
            new ProcessRequest(
                executable,
                arguments,
                _processTimeout,
                WorkingDirectory: _layout.Root,
                Environment: environment),
            cancellationToken).ConfigureAwait(false);
        if (result.TimedOut)
        {
            throw new ProcessTimeoutException($"{stage} exceeded {_processTimeout.TotalSeconds:F0} seconds");
        }

        if (!result.Succeeded)
        {
            string diagnostic = ProcessDiagnostic(result);
            throw new DependencyInstallException($"{stage} failed: {diagnostic}");
        }

        return result;
    }

    internal static JsonElement ParseDoctor(ProcessResult result, BackendProfile expected)
    {
        if (result.TimedOut)
        {
            throw new ProcessTimeoutException("installed swarm node doctor timed out");
        }

        if (!result.Succeeded)
        {
            throw new DoctorFailureException(
                $"installed swarm node doctor failed: {ProcessDiagnostic(result)}");
        }

        try
        {
            using JsonDocument document = JsonDocument.Parse(result.StandardOutput);
            JsonElement root = document.RootElement;
            if (root.GetProperty("status").GetString() != "pass")
            {
                throw new DoctorFailureException("installed doctor did not report pass");
            }

            JsonElement selection = root.GetProperty("backend_selection");
            string selected = selection.GetProperty("selected_backend").GetString() ?? string.Empty;
            string required = expected == BackendProfile.Cuda ? "torch-cuda" : "torch-cpu";
            if (!string.Equals(selected, required, StringComparison.Ordinal))
            {
                throw new DoctorFailureException(
                    $"{expected.ToString().ToLowerInvariant()} profile selected {selected}, not {required}");
            }

            if (expected == BackendProfile.Cuda && !OperationalCudaCandidate(selection))
            {
                throw new DoctorFailureException(
                    "CUDA profile did not prove an operational torch-cuda tensor candidate");
            }

            return root.Clone();
        }
        catch (JsonException exception)
        {
            throw new DoctorFailureException("installed doctor returned malformed JSON", exception);
        }
        catch (KeyNotFoundException exception)
        {
            throw new DoctorFailureException("installed doctor JSON is missing required fields", exception);
        }
    }

    private static bool OperationalCudaCandidate(JsonElement selection)
    {
        if (!selection.TryGetProperty("candidates", out JsonElement candidates)
            || candidates.ValueKind != JsonValueKind.Array)
        {
            return false;
        }

        foreach (JsonElement candidate in candidates.EnumerateArray())
        {
            if (candidate.TryGetProperty("backend", out JsonElement backend)
                && backend.GetString() == "torch-cuda"
                && candidate.TryGetProperty("operational", out JsonElement operational)
                && operational.ValueKind == JsonValueKind.True)
            {
                return true;
            }
        }

        return false;
    }

    private void CacheVerifiedPayload(ReleaseManifest manifest)
    {
        Directory.CreateDirectory(_layout.PayloadCache);
        List<FileAsset> assets =
        [
            manifest.Uv,
            manifest.Wheel,
            manifest.RuntimeProfiles.Cpu,
            manifest.RuntimeProfiles.Cuda,
            manifest.Bootstrapper,
            .. manifest.Payload,
        ];
        foreach (FileAsset asset in assets)
        {
            CopyVerified(
                Path.Combine(_payloadDirectory, asset.Filename),
                Path.Combine(_layout.PayloadCache, asset.Filename),
                asset);
        }

        string manifestSource = Path.Combine(_payloadDirectory, "release-manifest.json");
        byte[] manifestBytes = File.ReadAllBytes(manifestSource);
        AtomicFile.WriteAllBytes(Path.Combine(_layout.PayloadCache, "release-manifest.json"), manifestBytes);
        AtomicFile.WriteAllBytes(_layout.Manifest, manifestBytes);
        CopyVerified(
            Path.Combine(_payloadDirectory, manifest.Bootstrapper.Filename),
            Path.Combine(_layout.Bin, manifest.Bootstrapper.Filename),
            manifest.Bootstrapper);
        FileAsset icon = manifest.Payload.Single(asset => asset.Filename == "swarm.ico");
        CopyVerified(
            Path.Combine(_payloadDirectory, icon.Filename),
            Path.Combine(_layout.App, icon.Filename),
            icon);
    }

    private static void CopyVerified(string source, string destination, FileAsset asset)
    {
        HashVerifier.Verify(source, asset);
        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        string temporary = destination + $".{Guid.NewGuid():N}.tmp";
        try
        {
            File.Copy(source, temporary, overwrite: false);
            HashVerifier.Verify(temporary, asset);
            File.Move(temporary, destination, overwrite: true);
        }
        finally
        {
            File.Delete(temporary);
        }
    }

    private InstallRecord BuildInstallRecord(
        string operation,
        ReleaseManifest manifest,
        BackendProfile candidate,
        CandidateRuntime selected,
        string? rejected,
        string? reason,
        InstallRecord? previous)
    {
        return new InstallRecord
        {
            ProductVersion = manifest.Version,
            GitTag = manifest.GitTag,
            GitCommit = manifest.GitCommit,
            InstallationOperation = operation,
            CandidateBackend = candidate.ToString().ToLowerInvariant(),
            SelectedBackend = selected.Backend.ToString().ToLowerInvariant(),
            RejectedBackend = rejected,
            RejectionReason = reason,
            PythonVersion = manifest.Python.Version,
            UvVersion = manifest.Uv.Version,
            UvSha256 = manifest.Uv.Sha256,
            WheelFilename = manifest.Wheel.Filename,
            WheelSha256 = manifest.Wheel.Sha256,
            RuntimeProfileFilename = selected.ProfileFilename,
            RuntimeProfileSha256 = selected.ProfileSha256,
            InstalledAtUtc = DateTimeOffset.UtcNow.ToString("O"),
            InstallerVersion = manifest.Version,
            SignatureStatus = manifest.Installer.SignatureStatus,
            PreviousInstalledVersion = previous?.ProductVersion,
            DoctorSummary = selected.Doctor,
            ApplicationPath = _layout.Root,
            StatePath = _layout.StateRoot,
            ReleaseManifestSha256 = HashVerifier.ComputeSha256(
                Path.Combine(_payloadDirectory, "release-manifest.json")),
        };
    }

    private static string ProcessDiagnostic(ProcessResult result)
    {
        string raw = string.IsNullOrWhiteSpace(result.StandardError)
            ? result.StandardOutput
            : result.StandardError;
        string redacted = BootstrapLogger.Redact(raw.Trim());
        return redacted[..Math.Min(2000, redacted.Length)];
    }

    internal static void DeleteOwnedDirectory(string directory, string root)
    {
        if (!PathInfo.IsSameOrChild(directory, root)
            || string.Equals(
                Path.TrimEndingDirectorySeparator(directory),
                Path.TrimEndingDirectorySeparator(root),
                StringComparison.OrdinalIgnoreCase))
        {
            throw new PermissionFailureException($"refusing to remove unowned path {directory}");
        }

        if (Directory.Exists(directory))
        {
            DeleteDirectoryWithoutFollowingReparsePoints(directory);
        }
    }

    private static void DeleteDirectoryWithoutFollowingReparsePoints(string directory)
    {
        FileAttributes directoryAttributes = File.GetAttributes(directory);
        if ((directoryAttributes & FileAttributes.ReparsePoint) != 0)
        {
            Directory.Delete(directory, recursive: false);
            return;
        }

        foreach (string entry in Directory.EnumerateFileSystemEntries(directory).ToArray())
        {
            FileAttributes attributes;
            try
            {
                attributes = File.GetAttributes(entry);
            }
            catch (FileNotFoundException)
            {
                continue;
            }
            catch (DirectoryNotFoundException)
            {
                continue;
            }

            if ((attributes & FileAttributes.Directory) != 0)
            {
                if ((attributes & FileAttributes.ReadOnly) != 0)
                {
                    File.SetAttributes(entry, attributes & ~FileAttributes.ReadOnly);
                }

                DeleteDirectoryWithoutFollowingReparsePoints(entry);
            }
            else
            {
                if ((attributes & FileAttributes.ReadOnly) != 0)
                {
                    File.SetAttributes(entry, attributes & ~FileAttributes.ReadOnly);
                }

                File.Delete(entry);
            }
        }

        directoryAttributes = File.GetAttributes(directory);
        if ((directoryAttributes & FileAttributes.ReadOnly) != 0)
        {
            File.SetAttributes(directory, directoryAttributes & ~FileAttributes.ReadOnly);
        }

        Directory.Delete(directory, recursive: false);
    }
}
