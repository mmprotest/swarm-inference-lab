using System.Text.Json;
using System.Text.RegularExpressions;

namespace SwarmBootstrap;

internal enum ExitCode
{
    Success = 0,
    InvalidArguments = 2,
    UnsupportedPlatform = 10,
    ManifestFailure = 20,
    HashFailure = 21,
    DependencyInstallationFailure = 30,
    DoctorFailure = 40,
    UpgradeRollback = 50,
    ServiceLifecycleFailure = 60,
    PermissionFailure = 70,
    Timeout = 80,
    UnexpectedFailure = 99,
}

internal abstract class BootstrapException(string message, Exception? innerException = null)
    : Exception(message, innerException)
{
    public abstract ExitCode ExitCode { get; }
}

internal sealed class InvalidArgumentsException(string message, Exception? innerException = null)
    : BootstrapException(message, innerException)
{
    public override ExitCode ExitCode => ExitCode.InvalidArguments;
}

internal sealed class UnsupportedPlatformException(string message, Exception? innerException = null)
    : BootstrapException(message, innerException)
{
    public override ExitCode ExitCode => ExitCode.UnsupportedPlatform;
}

internal sealed class ManifestException(string message, Exception? innerException = null)
    : BootstrapException(message, innerException)
{
    public override ExitCode ExitCode => ExitCode.ManifestFailure;
}

internal sealed class HashMismatchException(string message, Exception? innerException = null)
    : BootstrapException(message, innerException)
{
    public override ExitCode ExitCode => ExitCode.HashFailure;
}

internal sealed class DependencyInstallException(string message, Exception? innerException = null)
    : BootstrapException(message, innerException)
{
    public override ExitCode ExitCode => ExitCode.DependencyInstallationFailure;
}

internal sealed class DoctorFailureException(string message, Exception? innerException = null)
    : BootstrapException(message, innerException)
{
    public override ExitCode ExitCode => ExitCode.DoctorFailure;
}

internal sealed class UpgradeRollbackException(string message, Exception? innerException = null)
    : BootstrapException(message, innerException)
{
    public override ExitCode ExitCode => ExitCode.UpgradeRollback;
}

internal sealed class ServiceLifecycleException(string message, Exception? innerException = null)
    : BootstrapException(message, innerException)
{
    public override ExitCode ExitCode => ExitCode.ServiceLifecycleFailure;
}

internal sealed class PermissionFailureException(string message, Exception? innerException = null)
    : BootstrapException(message, innerException)
{
    public override ExitCode ExitCode => ExitCode.PermissionFailure;
}

internal sealed class ProcessTimeoutException(string message, Exception? innerException = null)
    : BootstrapException(message, innerException)
{
    public override ExitCode ExitCode => ExitCode.Timeout;
}

internal static class JsonOutput
{
    public static void WriteSuccess(object? result, string operation, string logPath)
    {
        Console.Out.WriteLine(JsonSerializer.Serialize(
            new
            {
                status = "ok",
                exit_code = (int)ExitCode.Success,
                category = "success",
                operation,
                result,
                log_path = logPath,
            },
            JsonDefaults.Strict));
    }

    public static void WriteFailure(ExitCode exitCode, string operation, string message, string logPath)
    {
        Console.Out.WriteLine(JsonSerializer.Serialize(
            new
            {
                status = "error",
                exit_code = (int)exitCode,
                category = ToCategory(exitCode),
                operation,
                error = BootstrapLogger.Redact(message),
                log_path = logPath,
            },
            JsonDefaults.Strict));
    }

    public static string ToCategory(ExitCode code) => code switch
    {
        ExitCode.Success => "success",
        ExitCode.InvalidArguments => "invalid-arguments",
        ExitCode.UnsupportedPlatform => "unsupported-platform",
        ExitCode.ManifestFailure => "manifest-failure",
        ExitCode.HashFailure => "hash-failure",
        ExitCode.DependencyInstallationFailure => "dependency-installation-failure",
        ExitCode.DoctorFailure => "doctor-failure",
        ExitCode.UpgradeRollback => "upgrade-rollback",
        ExitCode.ServiceLifecycleFailure => "service-lifecycle-failure",
        ExitCode.PermissionFailure => "permission-failure",
        ExitCode.Timeout => "timeout",
        _ => "unexpected-failure",
    };
}

internal sealed partial class BootstrapLogger : IDisposable
{
    private readonly object _gate = new();
    private readonly StreamWriter _writer;
    private readonly bool _console;

    public BootstrapLogger(string path, bool console)
    {
        LogPath = Path.GetFullPath(path);
        string directory = Path.GetDirectoryName(LogPath)
            ?? throw new IOException("log path has no parent directory");
        Directory.CreateDirectory(directory);
        FileStream stream = new(LogPath, FileMode.Append, FileAccess.Write, FileShare.Read);
        _writer = new StreamWriter(stream, new System.Text.UTF8Encoding(false)) { AutoFlush = true };
        _console = console;
    }

    public string LogPath { get; }

    public void Info(string value) => Write("INFO", value);

    public void Diagnostic(string value) => Write("DIAGNOSTIC", value);

    public void Error(string value) => Write("ERROR", value);

    public void Dispose() => _writer.Dispose();

    public static string Redact(string value)
    {
        string result = PairingUriPattern().Replace(value, "<redacted-pairing-uri>");
        result = GithubTokenPattern().Replace(result, "<redacted-github-token>");
        result = SensitiveAssignmentPattern().Replace(result, "$1=<redacted>");
        return result;
    }

    private void Write(string level, string value)
    {
        string line = $"{DateTimeOffset.UtcNow:O} {level} {Redact(value)}";
        lock (_gate)
        {
            _writer.WriteLine(line);
        }

        if (_console)
        {
            Console.Error.WriteLine(line);
        }
    }

    [GeneratedRegex(@"swarm(?:\+pair)?://[^\s\""']+", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex PairingUriPattern();

    [GeneratedRegex(@"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b", RegexOptions.CultureInvariant)]
    private static partial Regex GithubTokenPattern();

    [GeneratedRegex(@"(?i)\b(token|password|secret|pfx)\s*=\s*[^\s;]+", RegexOptions.CultureInvariant)]
    private static partial Regex SensitiveAssignmentPattern();
}
