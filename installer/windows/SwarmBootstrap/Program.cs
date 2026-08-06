using System.Text.Json;

namespace SwarmBootstrap;

internal static class Program
{
    private static readonly HashSet<string> Operations =
    [
        "install",
        "repair",
        "upgrade",
        "uninstall",
        "doctor",
        "detect-backend",
    ];

    public static async Task<int> Main(string[] args)
    {
        CommandLine? command = null;
        BootstrapLogger? logger = null;
        try
        {
            command = CommandLine.Parse(args);
            string logPath = command.LogPath ?? DefaultLogPath(command.InstallRoot);
            logger = new BootstrapLogger(logPath, console: !command.Json);
            logger.Info($"bootstrapper start operation={command.Operation}");
            object? result = await ExecuteAsync(command, logger).ConfigureAwait(false);
            logger.Info($"bootstrapper success operation={command.Operation}");
            if (command.Json)
            {
                JsonOutput.WriteSuccess(result, command.Operation, logger.LogPath);
            }
            else
            {
                WriteHumanSuccess(command.Operation, result, logger.LogPath);
            }

            return (int)ExitCode.Success;
        }
        catch (Exception exception)
        {
            ExitCode exitCode = Classify(exception);
            string operation = command?.Operation ?? "arguments";
            string logPath = logger?.LogPath ?? string.Empty;
            logger?.Error($"bootstrapper failure category={JsonOutput.ToCategory(exitCode)}: {exception}");
            bool json = command?.Json ?? args.Contains("--json", StringComparer.OrdinalIgnoreCase);
            if (json)
            {
                JsonOutput.WriteFailure(exitCode, operation, exception.Message, logPath);
            }
            else
            {
                Console.Error.WriteLine(
                    $"Swarm Inference setup failed ({JsonOutput.ToCategory(exitCode)}): "
                    + BootstrapLogger.Redact(exception.Message));
                if (logPath.Length > 0)
                {
                    Console.Error.WriteLine($"Log: {logPath}");
                }
            }

            return (int)exitCode;
        }
        finally
        {
            logger?.Dispose();
        }
    }

    private static async Task<object?> ExecuteAsync(CommandLine command, BootstrapLogger logger)
    {
        using CancellationTokenSource timeout = new(TimeSpan.FromSeconds(command.TimeoutSeconds + 30));
        BoundedProcess processRunner = new(logger);
        if (command.Operation == "detect-backend")
        {
            BackendDetectionResult detection = await new BackendDetector(processRunner, logger)
                .DetectAsync(timeout.Token).ConfigureAwait(false);
            return new
            {
                candidate = detection.Candidate.ToString().ToLowerInvariant(),
                nvidia_probe_succeeded = detection.NvidiaProbeSucceeded,
                reason = detection.Reason,
                probe = detection.Probe is null
                    ? null
                    : new
                    {
                        exit_code = detection.Probe.ExitCode,
                        timed_out = detection.Probe.TimedOut,
                        duration_ms = detection.Probe.Duration.TotalMilliseconds,
                    },
            };
        }

        PathInfo layout = PathInfo.Create(command.InstallRoot!);
        RuntimeInstaller installer = new(
            layout,
            command.Payload ?? layout.PayloadCache,
            processRunner,
            logger,
            TimeSpan.FromSeconds(command.TimeoutSeconds),
            command.KeepFailedStaging);
        if (command.Operation == "doctor")
        {
            return await installer.DoctorAsync(timeout.Token).ConfigureAwait(false);
        }

        if (command.Operation == "uninstall")
        {
            await installer.UninstallAsync(command.PurgeState, timeout.Token).ConfigureAwait(false);
            return new
            {
                install_root = layout.Root,
                state_root = layout.StateRoot,
                state_preserved = !command.PurgeState,
            };
        }

        return await installer.ExecuteAsync(
                command.Operation,
                command.Backend,
                command.AllowDowngrade,
                timeout.Token)
            .ConfigureAwait(false);
    }

    private static ExitCode Classify(Exception exception)
    {
        if (exception is BootstrapException bootstrap)
        {
            return bootstrap.ExitCode;
        }

        return exception switch
        {
            ArgumentException or FormatException => ExitCode.InvalidArguments,
            UnauthorizedAccessException => ExitCode.PermissionFailure,
            OperationCanceledException => ExitCode.Timeout,
            _ => ExitCode.UnexpectedFailure,
        };
    }

    private static string DefaultLogPath(string? installRoot)
    {
        string root = installRoot is null
            ? Path.Combine(
                Environment.GetEnvironmentVariable("LOCALAPPDATA")
                    ?? Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "SwarmInference",
                "installer-logs")
            : Path.Combine(Path.GetFullPath(installRoot), "logs");
        return Path.Combine(root, $"bootstrapper-{DateTime.UtcNow:yyyyMMdd-HHmmss}-{Environment.ProcessId}.log");
    }

    private static void WriteHumanSuccess(string operation, object? result, string logPath)
    {
        if (result is InstallationResult installation)
        {
            Console.WriteLine(
                $"Swarm Inference {installation.Version} installed with the "
                + $"{installation.SelectedProfile} profile.");
            if (installation.RejectedProfile is not null)
            {
                Console.WriteLine(
                    $"The {installation.RejectedProfile} candidate was rejected: "
                    + installation.RejectionReason);
            }

            Console.WriteLine("Open a new terminal, then run: swarm node doctor");
        }
        else if (operation == "uninstall")
        {
            Console.WriteLine("Swarm Inference application files were removed.");
        }
        else
        {
            Console.WriteLine(JsonSerializer.Serialize(result, JsonDefaults.Strict));
        }

        Console.WriteLine($"Log: {logPath}");
    }
}

internal sealed record CommandLine(
    string Operation,
    string? Payload,
    string? InstallRoot,
    bool Json,
    string? LogPath,
    int TimeoutSeconds,
    BackendProfile Backend,
    bool KeepFailedStaging,
    bool PurgeState,
    bool AllowDowngrade)
{
    public static CommandLine Parse(IReadOnlyList<string> arguments)
    {
        if (arguments.Count == 0 || !Operations.Contains(arguments[0]))
        {
            throw new InvalidArgumentsException(Usage());
        }

        string operation = arguments[0];
        string? payload = null;
        string? installRoot = null;
        string? logPath = null;
        bool json = false;
        bool keepFailedStaging = false;
        bool purgeState = false;
        bool allowDowngrade = false;
        int timeoutSeconds = 900;
        BackendProfile backend = BackendProfile.Auto;
        for (int index = 1; index < arguments.Count; index++)
        {
            string option = arguments[index];
            switch (option)
            {
                case "--json":
                    json = true;
                    break;
                case "--keep-failed-staging":
                    keepFailedStaging = true;
                    break;
                case "--purge-state":
                    purgeState = true;
                    break;
                case "--allow-downgrade":
                    allowDowngrade = true;
                    break;
                case "--payload":
                    payload = RequireValue(arguments, ref index, option);
                    break;
                case "--install-root":
                    installRoot = RequireValue(arguments, ref index, option);
                    break;
                case "--log":
                    logPath = RequireValue(arguments, ref index, option);
                    break;
                case "--timeout-seconds":
                    string timeoutValue = RequireValue(arguments, ref index, option);
                    if (!int.TryParse(timeoutValue, out timeoutSeconds)
                        || timeoutSeconds is < 30 or > 3600)
                    {
                        throw new InvalidArgumentsException(
                            "--timeout-seconds must be an integer from 30 through 3600");
                    }

                    break;
                case "--backend":
                    string backendValue = RequireValue(arguments, ref index, option);
                    backend = backendValue.ToLowerInvariant() switch
                    {
                        "auto" => BackendProfile.Auto,
                        "cpu" => BackendProfile.Cpu,
                        "cuda" => BackendProfile.Cuda,
                        _ => throw new InvalidArgumentsException("--backend must be auto, cpu, or cuda"),
                    };
                    break;
                default:
                    throw new InvalidArgumentsException($"unknown option: {option}\n{Usage()}");
            }
        }

        bool mutating = operation is "install" or "repair" or "upgrade";
        if (operation != "detect-backend" && string.IsNullOrWhiteSpace(installRoot))
        {
            throw new InvalidArgumentsException($"{operation} requires --install-root");
        }

        if (mutating && string.IsNullOrWhiteSpace(payload))
        {
            throw new InvalidArgumentsException($"{operation} requires --payload");
        }

        if (!mutating && payload is not null)
        {
            throw new InvalidArgumentsException($"{operation} does not accept --payload");
        }

        if (operation is "doctor" or "detect-backend" && (purgeState || allowDowngrade))
        {
            throw new InvalidArgumentsException($"{operation} does not accept mutating recovery flags");
        }

        if (operation != "uninstall" && purgeState)
        {
            throw new InvalidArgumentsException("--purge-state is valid only with uninstall");
        }

        if (operation is not ("install" or "upgrade") && allowDowngrade)
        {
            throw new InvalidArgumentsException(
                "--allow-downgrade is valid only with install or upgrade");
        }

        return new CommandLine(
            operation,
            payload,
            installRoot,
            json,
            logPath,
            timeoutSeconds,
            backend,
            keepFailedStaging,
            purgeState,
            allowDowngrade);
    }

    private static string RequireValue(IReadOnlyList<string> arguments, ref int index, string option)
    {
        index++;
        if (index >= arguments.Count || arguments[index].StartsWith("--", StringComparison.Ordinal))
        {
            throw new InvalidArgumentsException($"{option} requires a value");
        }

        return arguments[index];
    }

    private static string Usage() =>
        "Usage: SwarmBootstrap.exe install|repair|upgrade --payload DIR --install-root DIR "
        + "[--backend auto|cpu|cuda] [--json] [--log PATH] [--timeout-seconds N]\n"
        + "       SwarmBootstrap.exe uninstall|doctor --install-root DIR [--json]\n"
        + "       SwarmBootstrap.exe detect-backend --json";

    private static HashSet<string> Operations =>
    [
        "install",
        "repair",
        "upgrade",
        "uninstall",
        "doctor",
        "detect-backend",
    ];
}
