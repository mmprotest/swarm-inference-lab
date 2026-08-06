using System.Runtime.InteropServices;

namespace SwarmBootstrap;

internal sealed record BackendDetectionResult(
    BackendProfile Candidate,
    bool NvidiaProbeSucceeded,
    string Reason,
    string? NvidiaSmiPath,
    ProcessResult? Probe);

internal sealed class BackendDetector
{
    private readonly IBoundedProcessRunner _processes;
    private readonly BootstrapLogger _logger;
    private readonly Func<string, string?> _findExecutable;

    public BackendDetector(
        IBoundedProcessRunner processes,
        BootstrapLogger logger,
        Func<string, string?>? findExecutable = null)
    {
        _processes = processes;
        _logger = logger;
        _findExecutable = findExecutable ?? FindOnPath;
    }

    public static void ValidatePlatform()
    {
        ValidatePlatformInputs(
            OperatingSystem.IsWindows(),
            RuntimeInformation.OSArchitecture,
            Environment.Is64BitOperatingSystem,
            Environment.OSVersion.Version);
    }

    internal static void ValidatePlatformInputs(
        bool isWindows,
        Architecture architecture,
        bool is64BitOperatingSystem,
        Version operatingSystemVersion)
    {
        if (!isWindows)
        {
            throw new UnsupportedPlatformException("Swarm Inference native setup requires Windows");
        }

        if (architecture != Architecture.X64 || !is64BitOperatingSystem)
        {
            throw new UnsupportedPlatformException("Swarm Inference native setup requires Windows x86-64");
        }

        Version minimum = new(10, 0, 22621);
        if (operatingSystemVersion < minimum)
        {
            throw new UnsupportedPlatformException(
                $"Windows {minimum} or newer is required; detected {operatingSystemVersion}");
        }
    }

    public async Task<BackendDetectionResult> DetectAsync(CancellationToken cancellationToken)
    {
        ValidatePlatform();
        string? executable = _findExecutable("nvidia-smi.exe");
        if (executable is null)
        {
            return new BackendDetectionResult(
                BackendProfile.Cpu,
                false,
                "nvidia-smi was not found on the bounded process PATH",
                null,
                null);
        }

        ProcessResult result = await _processes.RunAsync(
            new ProcessRequest(
                executable,
                ["--query-gpu=name,driver_version", "--format=csv,noheader"],
                TimeSpan.FromSeconds(20)),
            cancellationToken).ConfigureAwait(false);
        if (!result.Succeeded || string.IsNullOrWhiteSpace(result.StandardOutput))
        {
            string detail = result.TimedOut
                ? "nvidia-smi timed out"
                : (result.StandardError.Trim().Length > 0
                    ? result.StandardError.Trim()
                    : $"nvidia-smi exited {result.ExitCode}");
            return new BackendDetectionResult(
                BackendProfile.Cpu,
                false,
                BootstrapLogger.Redact(detail),
                executable,
                result);
        }

        _logger.Info("NVIDIA driver probe succeeded; CUDA is a candidate pending installed doctor validation");
        return new BackendDetectionResult(
            BackendProfile.Cuda,
            true,
            "nvidia-smi reported a functioning NVIDIA driver; CUDA remains provisional",
            executable,
            result);
    }

    internal static string? FindOnPath(string filename)
    {
        if (Path.IsPathFullyQualified(filename) && File.Exists(filename))
        {
            return Path.GetFullPath(filename);
        }

        string? pathValue = Environment.GetEnvironmentVariable("PATH");
        if (string.IsNullOrWhiteSpace(pathValue))
        {
            return null;
        }

        foreach (string raw in pathValue.Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            string directory = Environment.ExpandEnvironmentVariables(raw.Trim().Trim('"'));
            if (directory.Length == 0)
            {
                continue;
            }

            try
            {
                string candidate = Path.Combine(directory, filename);
                if (File.Exists(candidate))
                {
                    return Path.GetFullPath(candidate);
                }
            }
            catch (Exception exception) when (
                exception is ArgumentException or NotSupportedException or PathTooLongException)
            {
                // Ignore a malformed unrelated PATH entry.
            }
        }

        return null;
    }
}
