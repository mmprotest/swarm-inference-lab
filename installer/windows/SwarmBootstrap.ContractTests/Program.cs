using System.Diagnostics;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Text.Json;
using SwarmBootstrap;

namespace SwarmBootstrap.ContractTests;

internal static class Program
{
    private static int _passed;

    public static async Task<int> Main(string[] arguments)
    {
        if (arguments is ["--helper-sleep"])
        {
            await Task.Delay(TimeSpan.FromMinutes(5)).ConfigureAwait(false);
            return 0;
        }

        if (arguments is ["--helper-tree", string pidFile])
        {
            using Process child = Process.Start(new ProcessStartInfo
            {
                FileName = Environment.ProcessPath!,
                UseShellExecute = false,
                ArgumentList = { "--helper-sleep" },
            })!;
            await File.WriteAllTextAsync(pidFile, child.Id.ToString()).ConfigureAwait(false);
            await child.WaitForExitAsync().ConfigureAwait(false);
            return child.ExitCode;
        }

        List<(string Name, Func<Task> Test)> tests =
        [
            ("stable-exit-codes", TestExitCodesAsync),
            ("platform-rejection", TestPlatformRejectionAsync),
            ("backend-detection", TestBackendDetectionAsync),
            ("explicit-cuda-failure", TestExplicitCudaFailureAsync),
            ("automatic-cuda-fallback", TestAutomaticCudaFallbackAsync),
            ("bounded-process-timeout", TestBoundedTimeoutAsync),
            ("process-tree-termination", TestProcessTreeTerminationAsync),
            ("strict-manifest-parsing", TestStrictManifestParsingAsync),
            ("hash-mismatch-rejection", TestHashMismatchAsync),
            ("engine-runtime-safe-extraction", TestEngineRuntimeExtractionAsync),
            ("engine-runtime-traversal-rejection", TestEngineRuntimeTraversalAsync),
            ("external-manifest-setup-binding", TestExternalManifestBindingAsync),
            ("external-manifest-redirect-policy", TestReleaseManifestRedirectPolicyAsync),
            ("atomic-install", TestAtomicInstallAsync),
            ("upgrade-commit", TestUpgradeCommitAsync),
            ("rollback-restoration", TestRollbackAsync),
            ("payload-cache-rollback", TestPayloadCacheRollbackAsync),
            ("payload-cache-commit-cleanup", TestPayloadCacheCommitCleanupAsync),
            ("path-idempotency", TestPathIdempotencyAsync),
            ("install-record-atomicity", TestInstallRecordAtomicityAsync),
            ("owned-tree-reparse-safe-delete", TestOwnedTreeDeleteAsync),
            ("output-redaction", TestOutputRedactionAsync),
            ("doctor-contract", TestDoctorContractAsync),
        ];
        List<object> failures = [];
        foreach ((string name, Func<Task> test) in tests)
        {
            try
            {
                await test().ConfigureAwait(false);
                _passed++;
            }
            catch (Exception exception)
            {
                failures.Add(new { name, error = exception.ToString() });
            }
        }

        Console.WriteLine(JsonSerializer.Serialize(new
        {
            status = failures.Count == 0 ? "PASS" : "FAIL",
            passed = _passed,
            failed = failures.Count,
            failures,
        }, JsonDefaults.Strict));
        return failures.Count == 0 ? 0 : 1;
    }

    private static Task TestExitCodesAsync()
    {
        int[] values = Enum.GetValues<ExitCode>().Select(value => (int)value).ToArray();
        Equal(values.Length, values.Distinct().Count(), "exit codes must be unique");
        Equal(50, (int)ExitCode.UpgradeRollback, "rollback code changed");
        Equal(80, (int)ExitCode.Timeout, "timeout code changed");
        return Task.CompletedTask;
    }

    private static Task TestPlatformRejectionAsync()
    {
        Throws<UnsupportedPlatformException>(() => BackendDetector.ValidatePlatformInputs(
            false, Architecture.X64, true, new Version(10, 0, 22621)));
        Throws<UnsupportedPlatformException>(() => BackendDetector.ValidatePlatformInputs(
            true, Architecture.Arm64, true, new Version(10, 0, 22621)));
        Throws<UnsupportedPlatformException>(() => BackendDetector.ValidatePlatformInputs(
            true, Architecture.X64, true, new Version(10, 0, 19045)));
        return Task.CompletedTask;
    }

    private static async Task TestBackendDetectionAsync()
    {
        using TestArea area = new();
        using BootstrapLogger logger = new(area.Path("backend.log"), console: false);
        FakeRunner runner = new(new ProcessResult(0, "RTX 5090, 600.1\n", "", false, TimeSpan.Zero));
        BackendDetectionResult result = await new BackendDetector(
            runner,
            logger,
            _ => @"C:\Windows\System32\nvidia-smi.exe").DetectAsync(CancellationToken.None);
        Equal(BackendProfile.Cuda, result.Candidate, "successful probe must be provisional CUDA");
        True(result.NvidiaProbeSucceeded, "probe success was not recorded");
    }

    private static Task TestExplicitCudaFailureAsync()
    {
        string json = DoctorJson("torch-cpu", cudaOperational: false);
        Throws<DoctorFailureException>(() => RuntimeInstaller.ParseDoctor(
            new ProcessResult(0, json, "", false, TimeSpan.Zero), BackendProfile.Cuda));
        EqualSequence(
            new[] { BackendProfile.Cuda },
            RuntimeInstaller.CandidateOrder(
                BackendProfile.Cuda,
                new BackendDetectionResult(BackendProfile.Cuda, true, "explicit", null, null)),
            "explicit CUDA must not include CPU fallback");
        return Task.CompletedTask;
    }

    private static Task TestAutomaticCudaFallbackAsync()
    {
        EqualSequence(
            new[] { BackendProfile.Cuda, BackendProfile.Cpu },
            RuntimeInstaller.CandidateOrder(
                BackendProfile.Auto,
                new BackendDetectionResult(BackendProfile.Cuda, true, "probe", null, null)),
            "automatic CUDA candidate must be followed by CPU fallback");
        return Task.CompletedTask;
    }

    private static async Task TestBoundedTimeoutAsync()
    {
        using TestArea area = new();
        using BootstrapLogger logger = new(area.Path("timeout.log"), console: false);
        ProcessResult result = await new BoundedProcess(logger).RunAsync(
            new ProcessRequest(
                Environment.ProcessPath!,
                ["--helper-sleep"],
                TimeSpan.FromMilliseconds(250)),
            CancellationToken.None);
        True(result.TimedOut, "long-running process was not timed out");
        True(result.Duration < TimeSpan.FromSeconds(15), "timeout was not bounded");
    }

    private static async Task TestProcessTreeTerminationAsync()
    {
        using TestArea area = new();
        using BootstrapLogger logger = new(area.Path("tree.log"), console: false);
        string pidFile = area.Path("child.pid");
        ProcessResult result = await new BoundedProcess(logger).RunAsync(
            new ProcessRequest(
                Environment.ProcessPath!,
                ["--helper-tree", pidFile],
                TimeSpan.FromSeconds(1)),
            CancellationToken.None);
        True(result.TimedOut, "process tree fixture was not timed out");
        True(File.Exists(pidFile), "child process identity was not recorded");
        int childId = int.Parse(await File.ReadAllTextAsync(pidFile).ConfigureAwait(false));
        try
        {
            using Process child = Process.GetProcessById(childId);
            True(child.WaitForExit(5000), "child survived entire-process-tree termination");
        }
        catch (ArgumentException)
        {
            // The child is already gone.
        }
    }

    private static Task TestStrictManifestParsingAsync()
    {
        string malformed = "{\"schema_version\":1,\"unexpected\":true}";
        Throws<JsonException>(() => JsonSerializer.Deserialize<ReleaseManifest>(malformed, JsonDefaults.Strict));
        return Task.CompletedTask;
    }

    private static async Task TestHashMismatchAsync()
    {
        using TestArea area = new();
        string file = area.Path("asset.bin");
        await File.WriteAllTextAsync(file, "actual").ConfigureAwait(false);
        FileAsset asset = new()
        {
            Filename = "asset.bin",
            Sha256 = "sha256:" + new string('0', 64),
            SizeBytes = new FileInfo(file).Length,
        };
        Throws<HashMismatchException>(() => HashVerifier.Verify(file, asset));
    }

    private static Task TestEngineRuntimeExtractionAsync()
    {
        using TestArea area = new();
        string archivePath = area.Path("runtime.zip");
        using (ZipArchive archive = ZipFile.Open(archivePath, ZipArchiveMode.Create))
        {
            using (StreamWriter server = new(archive.CreateEntry("llama-server.exe").Open()))
            {
                server.Write("server");
            }
            using (StreamWriter rpc = new(archive.CreateEntry("rpc-server.exe").Open()))
            {
                rpc.Write("rpc");
            }
        }
        string destination = area.Path("runtime");
        Directory.CreateDirectory(destination);
        HashSet<string> files = new(StringComparer.OrdinalIgnoreCase);
        long bytes = 0;
        RuntimeInstaller.ExtractEngineArchive(archivePath, destination, files, ref bytes);
        True(File.Exists(Path.Combine(destination, "llama-server.exe")),
            "llama.cpp server was not extracted");
        True(File.Exists(Path.Combine(destination, "rpc-server.exe")),
            "llama.cpp RPC server was not extracted");
        Equal(9L, bytes, "llama.cpp extracted byte accounting is wrong");
        return Task.CompletedTask;
    }

    private static Task TestEngineRuntimeTraversalAsync()
    {
        using TestArea area = new();
        string archivePath = area.Path("unsafe.zip");
        using (ZipArchive archive = ZipFile.Open(archivePath, ZipArchiveMode.Create))
        {
            using StreamWriter writer = new(archive.CreateEntry("../escape.exe").Open());
            writer.Write("unsafe");
        }
        string destination = area.Path("runtime");
        Directory.CreateDirectory(destination);
        HashSet<string> files = new(StringComparer.OrdinalIgnoreCase);
        long bytes = 0;
        Throws<ManifestException>(() =>
            RuntimeInstaller.ExtractEngineArchive(
                archivePath,
                destination,
                files,
                ref bytes));
        True(!File.Exists(area.Path("escape.exe")), "unsafe ZIP member escaped the runtime root");
        return Task.CompletedTask;
    }

    private static async Task TestExternalManifestBindingAsync()
    {
        using TestArea area = new();
        string setup = area.Path("SwarmInferenceSetup-x64.exe");
        await File.WriteAllTextAsync(setup, "setup-fixture").ConfigureAwait(false);
        ReleaseManifest embedded = ManifestFixture("embedded-payload", setup: null);
        ReleaseManifest release = ManifestFixture("release", setup);
        byte[] content = JsonSerializer.SerializeToUtf8Bytes(release, JsonDefaults.Strict);
        ReleaseManifest validated = HashVerifier.ValidateReleaseManifest(content, embedded, setup);
        Equal("release", validated.ManifestScope, "external release manifest was rejected");
        byte[] mismatch = JsonSerializer.SerializeToUtf8Bytes(
            release with { GitCommit = new string('b', 40) },
            JsonDefaults.Strict);
        Throws<ManifestException>(() =>
            HashVerifier.ValidateReleaseManifest(mismatch, embedded, setup));
    }

    private static Task TestReleaseManifestRedirectPolicyAsync()
    {
        ReleaseManifestResolver.ValidateUri(new Uri("https://github.com/owner/release"));
        Throws<ManifestException>(() =>
            ReleaseManifestResolver.ValidateUri(new Uri("https://example.invalid/setup")));
        Throws<ManifestException>(() =>
            ReleaseManifestResolver.ValidateUri(new Uri("http://github.com/owner/release")));
        return Task.CompletedTask;
    }

    private static Task TestAtomicInstallAsync()
    {
        using TestArea area = new();
        PathInfo layout = PathInfo.Create(area.Path("app"));
        layout.EnsureBaseDirectories();
        using BootstrapLogger logger = new(area.Path("install.log"), console: false);
        using RuntimeTransaction transaction = new(layout, logger, keepFailedStaging: false);
        string staging = transaction.CreateStagingRuntime();
        File.WriteAllText(Path.Combine(staging, "candidate.txt"), "candidate");
        transaction.PublishStaging();
        True(File.Exists(Path.Combine(layout.Runtime, "candidate.txt")), "candidate was not published");
        transaction.Commit();
        return Task.CompletedTask;
    }

    private static Task TestUpgradeCommitAsync()
    {
        using TestArea area = new();
        PathInfo layout = PathInfo.Create(area.Path("app"));
        layout.EnsureBaseDirectories();
        Directory.CreateDirectory(layout.Runtime);
        File.WriteAllText(Path.Combine(layout.Runtime, "version.txt"), "A");
        using BootstrapLogger logger = new(area.Path("upgrade.log"), console: false);
        using RuntimeTransaction transaction = new(layout, logger, keepFailedStaging: false);
        transaction.MoveActiveAside(previous: null);
        string staging = transaction.CreateStagingRuntime();
        File.WriteAllText(Path.Combine(staging, "version.txt"), "B");
        transaction.PublishStaging();
        transaction.Commit();
        Equal("B", File.ReadAllText(Path.Combine(layout.Runtime, "version.txt")), "upgrade not committed");
        True(!Directory.EnumerateDirectories(layout.Previous).Any(), "previous runtime survived commit");
        return Task.CompletedTask;
    }

    private static Task TestRollbackAsync()
    {
        using TestArea area = new();
        PathInfo layout = PathInfo.Create(area.Path("app"));
        layout.EnsureBaseDirectories();
        Directory.CreateDirectory(layout.Runtime);
        File.WriteAllText(Path.Combine(layout.Runtime, "version.txt"), "A");
        AtomicFile.WriteAllText(layout.InstallRecord, "old-record");
        using BootstrapLogger logger = new(area.Path("rollback.log"), console: false);
        using RuntimeTransaction transaction = new(layout, logger, keepFailedStaging: false);
        transaction.MoveActiveAside(previous: null);
        string staging = transaction.CreateStagingRuntime();
        File.WriteAllText(Path.Combine(staging, "version.txt"), "broken-B");
        transaction.PublishStaging();
        AtomicFile.WriteAllText(layout.InstallRecord, "new-record");
        transaction.Rollback();
        Equal("A", File.ReadAllText(Path.Combine(layout.Runtime, "version.txt")), "runtime rollback failed");
        Equal("old-record", File.ReadAllText(layout.InstallRecord), "record rollback failed");
        return Task.CompletedTask;
    }

    private static Task TestPayloadCacheRollbackAsync()
    {
        using TestArea area = new();
        PathInfo layout = PathInfo.Create(area.Path("app"));
        layout.EnsureBaseDirectories();
        File.WriteAllText(Path.Combine(layout.PayloadCache, "version.txt"), "A");
        using BootstrapLogger logger = new(area.Path("payload-rollback.log"), console: false);
        using RuntimeTransaction transaction = new(layout, logger, keepFailedStaging: false);
        string staging = transaction.CreateStagingPayloadCache();
        File.WriteAllText(Path.Combine(staging, "version.txt"), "broken-B");
        transaction.PublishPayloadCache();
        transaction.Rollback();
        transaction.Rollback();
        Equal(
            "A",
            File.ReadAllText(Path.Combine(layout.PayloadCache, "version.txt")),
            "payload cache rollback failed");
        return Task.CompletedTask;
    }

    private static Task TestPayloadCacheCommitCleanupAsync()
    {
        using TestArea area = new();
        PathInfo layout = PathInfo.Create(area.Path("app"));
        layout.EnsureBaseDirectories();
        string retired = Path.Combine(layout.PayloadCache, "read-only.txt");
        File.WriteAllText(retired, "A");
        File.SetAttributes(retired, FileAttributes.ReadOnly);
        using BootstrapLogger logger = new(area.Path("payload-commit.log"), console: false);
        using RuntimeTransaction transaction = new(layout, logger, keepFailedStaging: false);
        string staging = transaction.CreateStagingPayloadCache();
        File.WriteAllText(Path.Combine(staging, "version.txt"), "B");
        transaction.PublishPayloadCache();
        transaction.Commit();
        Equal(
            "B",
            File.ReadAllText(Path.Combine(layout.PayloadCache, "version.txt")),
            "candidate payload cache was not committed");
        True(
            !Directory.EnumerateDirectories(layout.Previous).Any(),
            "retired payload cache survived commit");
        return Task.CompletedTask;
    }

    private static Task TestPathIdempotencyAsync()
    {
        string owned = Path.GetFullPath(@"C:\Apps\SwarmInference\runtime\Scripts");
        string unrelated = @"C:\Windows\System32";
        string once = UserPathRegistration.UpdatePathValue(unrelated, owned, add: true);
        string twice = UserPathRegistration.UpdatePathValue(once, owned + "\\", add: true);
        Equal(once, twice, "owned PATH entry was duplicated");
        Equal(unrelated, UserPathRegistration.UpdatePathValue(twice, owned, add: false),
            "PATH removal altered unrelated values");
        return Task.CompletedTask;
    }

    private static Task TestInstallRecordAtomicityAsync()
    {
        using TestArea area = new();
        PathInfo layout = PathInfo.Create(area.Path("app"));
        JsonElement doctor = JsonDocument.Parse(DoctorJson("torch-cpu", false)).RootElement.Clone();
        InstallRecord record = new()
        {
            ProductVersion = "0.1.0rc1",
            GitTag = "v0.1.0-rc.1",
            GitCommit = new string('a', 40),
            InstallationOperation = "install",
            CandidateBackend = "cpu",
            SelectedBackend = "cpu",
            PythonVersion = "3.11.15",
            UvVersion = "0.12.0",
            UvSha256 = "sha256:" + new string('1', 64),
            WheelFilename = "swarm.whl",
            WheelSha256 = "sha256:" + new string('2', 64),
            RuntimeProfileFilename = "cpu.lock",
            RuntimeProfileSha256 = "sha256:" + new string('3', 64),
            InstalledAtUtc = DateTimeOffset.UtcNow.ToString("O"),
            InstallerVersion = "0.1.0rc1",
            SignatureStatus = "unsigned-prerelease",
            DoctorSummary = doctor,
            ApplicationPath = layout.Root,
            StatePath = layout.StateRoot,
            ReleaseManifestSha256 = "sha256:" + new string('4', 64),
        };
        record.SaveAtomic(layout);
        InstallRecord loaded = InstallRecord.Load(layout)!;
        Equal(record.ProductVersion, loaded.ProductVersion, "atomic install record did not round trip");
        True(!Directory.EnumerateFiles(layout.App, "*.tmp").Any(), "atomic temp record was retained");
        return Task.CompletedTask;
    }

    private static Task TestOwnedTreeDeleteAsync()
    {
        using TestArea area = new();
        string root = area.Path("application");
        string owned = System.IO.Path.Combine(root, "runtime");
        string nested = System.IO.Path.Combine(owned, "nested");
        Directory.CreateDirectory(nested);
        File.WriteAllText(System.IO.Path.Combine(nested, "read-only.txt"), "owned");
        File.SetAttributes(
            System.IO.Path.Combine(nested, "read-only.txt"),
            FileAttributes.ReadOnly);
        RuntimeInstaller.DeleteOwnedDirectory(owned, root);
        True(!Directory.Exists(owned), "owned recursive tree was not removed");
        Throws<PermissionFailureException>(() => RuntimeInstaller.DeleteOwnedDirectory(root, root));
        return Task.CompletedTask;
    }

    private static Task TestOutputRedactionAsync()
    {
        string secret = "swarm://10.0.0.1:50051/join/c2VjcmV0";
        string redacted = BootstrapLogger.Redact($"join {secret} token=abc123 password=hunter2");
        True(!redacted.Contains("c2VjcmV0", StringComparison.Ordinal), "pairing URI was not redacted");
        True(!redacted.Contains("hunter2", StringComparison.Ordinal), "password was not redacted");
        ProcessRequest request = new("tool.exe", ["--token", "private"], TimeSpan.FromSeconds(1),
            SensitiveArgumentIndexes: new HashSet<int> { 1 });
        True(!BoundedProcess.RenderCommand(request).Contains("private", StringComparison.Ordinal),
            "sensitive process argument was not redacted");
        return Task.CompletedTask;
    }

    private static Task TestDoctorContractAsync()
    {
        JsonElement document = RuntimeInstaller.ParseDoctor(
            new ProcessResult(0, DoctorJson("torch-cpu", false), "", false, TimeSpan.Zero),
            BackendProfile.Cpu);
        Equal("pass", document.GetProperty("status").GetString(), "CPU doctor did not pass");
        JsonElement cuda = RuntimeInstaller.ParseDoctor(
            new ProcessResult(0, DoctorJson("torch-cuda", true), "", false, TimeSpan.Zero),
            BackendProfile.Cuda);
        Equal("torch-cuda", cuda.GetProperty("backend_selection").GetProperty("selected_backend").GetString(),
            "operational CUDA doctor was rejected");
        return Task.CompletedTask;
    }

    private static string DoctorJson(string selected, bool cudaOperational) => JsonSerializer.Serialize(new
    {
        status = "pass",
        backend_selection = new
        {
            selected_backend = selected,
            candidates = new[]
            {
                new { backend = "torch-cuda", operational = cudaOperational },
                new { backend = "torch-cpu", operational = true },
            },
        },
    });

    private static ReleaseManifest ManifestFixture(string scope, string? setup)
    {
        string hash = "sha256:" + new string('1', 64);
        FileAsset Asset(string filename) => new()
        {
            Filename = filename,
            Sha256 = hash,
            SizeBytes = 1,
        };
        LlamaCppRuntimeProfile LlamaProfile(bool cuda) => new()
        {
            Platform = "windows-x64",
            Archives = cuda
                ? [Asset("llama-cuda.zip"), Asset("cudart.zip")]
                : [Asset("llama-cpu.zip")],
            ServerBinary = "llama-server.exe",
            ServerSha256 = hash,
            RpcServerBinary = "rpc-server.exe",
            RpcServerSha256 = hash,
            BuildFlags = new Dictionary<string, bool>
            {
                ["GGML_CUDA"] = cuda,
                ["GGML_RPC"] = true,
            },
            DeviceSupport = cuda ? ["CPU", "CUDA"] : ["CPU"],
        };
        SignedAsset bootstrapper = new()
        {
            Filename = "SwarmBootstrap.exe",
            Sha256 = hash,
            SizeBytes = 1,
            SignatureStatus = "unsigned-prerelease",
            SignatureVerification = "not-signed",
        };
        SignedAsset installer = new()
        {
            Filename = "SwarmInferenceSetup-x64.exe",
            Sha256 = setup is null
                ? "sha256:" + new string('0', 64)
                : HashVerifier.ComputeSha256(setup),
            SizeBytes = setup is null ? 0 : new FileInfo(setup).Length,
            SignatureStatus = "unsigned-prerelease",
            SignatureVerification = "not-signed",
        };
        return new ReleaseManifest
        {
            SchemaVersion = 1,
            ManifestScope = scope,
            Product = "swarm-inference-lab",
            Version = "0.1.0rc1",
            GitTag = "v0.1.0-rc.1",
            GitCommit = new string('a', 40),
            Channel = "prerelease",
            BuiltAtUtc = "2026-08-06T00:00:00Z",
            MinimumWindows = "10.0.22621",
            Architecture = "x86_64",
            Python = new PythonAsset { Version = "3.11.15" },
            Uv = new UvAsset
            {
                Filename = "uv.exe",
                Sha256 = hash,
                SizeBytes = 1,
                Version = "0.12.0",
            },
            Wheel = Asset("swarm.whl"),
            RuntimeProfiles = new RuntimeProfiles
            {
                Cpu = Asset("cpu.lock"),
                Cuda = Asset("cuda.lock"),
            },
            EngineRuntimes = new EngineRuntimes
            {
                LlamaCpp = new LlamaCppRuntimeSet
                {
                    Repository = "ggml-org/llama.cpp",
                    ReleaseTag = "b9637",
                    RuntimeRevision = new string('b', 40),
                    Profiles = new LlamaCppRuntimeProfiles
                    {
                        Cpu = LlamaProfile(cuda: false),
                        Cuda = LlamaProfile(cuda: true),
                    },
                },
            },
            Bootstrapper = bootstrapper,
            Installer = installer,
            Payload = [],
        };
    }

    private static void True(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private static void Equal<T>(T expected, T actual, string message)
    {
        if (!EqualityComparer<T>.Default.Equals(expected, actual))
        {
            throw new InvalidOperationException($"{message}: expected={expected}, actual={actual}");
        }
    }

    private static void EqualSequence<T>(
        IEnumerable<T> expected,
        IEnumerable<T> actual,
        string message)
    {
        if (!expected.SequenceEqual(actual))
        {
            throw new InvalidOperationException(message);
        }
    }

    private static void Throws<TException>(Action action)
        where TException : Exception
    {
        try
        {
            action();
        }
        catch (TException)
        {
            return;
        }

        throw new InvalidOperationException($"expected {typeof(TException).Name}");
    }

    private sealed class FakeRunner(params ProcessResult[] results) : IBoundedProcessRunner
    {
        private readonly Queue<ProcessResult> _results = new(results);

        public Task<ProcessResult> RunAsync(
            ProcessRequest request,
            CancellationToken cancellationToken)
        {
            del(request, cancellationToken);
            return Task.FromResult(_results.Dequeue());
        }

        private static void del(params object[] values)
        {
            _ = values;
        }
    }

    private sealed class TestArea : IDisposable
    {
        private readonly string _root = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), $"swarm-bootstrap-tests-{Guid.NewGuid():N}");

        public TestArea() => Directory.CreateDirectory(_root);

        public string Path(string relative) => System.IO.Path.Combine(_root, relative);

        public void Dispose()
        {
            if (Directory.Exists(_root))
            {
                Directory.Delete(_root, recursive: true);
            }
        }
    }
}
