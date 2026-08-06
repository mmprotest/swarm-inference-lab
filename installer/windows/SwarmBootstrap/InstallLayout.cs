using Microsoft.Win32;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace SwarmBootstrap;

internal sealed record PathInfo
{
    public required string Root { get; init; }
    public required string App { get; init; }
    public required string Bin { get; init; }
    public required string Runtime { get; init; }
    public required string PayloadCache { get; init; }
    public required string Logs { get; init; }
    public required string Previous { get; init; }
    public required string StateRoot { get; init; }
    public required string Manifest { get; init; }
    public required string InstallRecord { get; init; }
    public required string SwarmExecutable { get; init; }
    public required string PythonExecutable { get; init; }
    public required string ScriptsPath { get; init; }

    public static PathInfo Create(string installRoot)
    {
        if (string.IsNullOrWhiteSpace(installRoot) || !Path.IsPathFullyQualified(installRoot))
        {
            throw new ArgumentException("--install-root must be an absolute path");
        }

        string root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(installRoot));
        string pathRoot = Path.GetPathRoot(root)
            ?? throw new ArgumentException("--install-root has no filesystem root");
        if (string.Equals(root, Path.TrimEndingDirectorySeparator(pathRoot), StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException("refusing a filesystem root as --install-root");
        }

        string localAppData = Environment.GetEnvironmentVariable("LOCALAPPDATA")
            ?? Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        string state = Path.Combine(Path.GetFullPath(localAppData), "SwarmInference");
        if (IsSameOrChild(state, root))
        {
            throw new ArgumentException("application installation and durable state roots must be separate");
        }

        string app = Path.Combine(root, "app");
        string runtime = Path.Combine(root, "runtime");
        string scripts = Path.Combine(runtime, "Scripts");
        return new PathInfo
        {
            Root = root,
            App = app,
            Bin = Path.Combine(root, "bin"),
            Runtime = runtime,
            PayloadCache = Path.Combine(root, "payload-cache"),
            Logs = Path.Combine(root, "logs"),
            Previous = Path.Combine(root, "previous"),
            StateRoot = state,
            Manifest = Path.Combine(app, "release-manifest.json"),
            InstallRecord = Path.Combine(app, "install-record.json"),
            ScriptsPath = scripts,
            SwarmExecutable = Path.Combine(scripts, "swarm.exe"),
            PythonExecutable = Path.Combine(scripts, "python.exe"),
        };
    }

    public void EnsureBaseDirectories()
    {
        foreach (string directory in new[] { Root, App, Bin, PayloadCache, Logs, Previous })
        {
            Directory.CreateDirectory(directory);
        }
    }

    public string NewStagingRuntime()
    {
        return Path.Combine(Root, $".runtime.{Guid.NewGuid():N}.staging");
    }

    internal static bool IsSameOrChild(string candidate, string parent)
    {
        string normalizedCandidate = Path.TrimEndingDirectorySeparator(Path.GetFullPath(candidate));
        string normalizedParent = Path.TrimEndingDirectorySeparator(Path.GetFullPath(parent));
        if (string.Equals(normalizedCandidate, normalizedParent, StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        return normalizedCandidate.StartsWith(
            normalizedParent + Path.DirectorySeparatorChar,
            StringComparison.OrdinalIgnoreCase);
    }
}

[SupportedOSPlatform("windows")]
internal static partial class UserPathRegistration
{
    private const string EnvironmentKey = @"Environment";

    public static void Add(string ownedPath)
    {
        using RegistryKey key = Registry.CurrentUser.CreateSubKey(EnvironmentKey, writable: true);
        string current = key.GetValue("Path", string.Empty, RegistryValueOptions.DoNotExpandEnvironmentNames)
            as string ?? string.Empty;
        string updated = UpdatePathValue(current, ownedPath, add: true);
        if (!string.Equals(current, updated, StringComparison.Ordinal))
        {
            key.SetValue("Path", updated, RegistryValueKind.ExpandString);
            BroadcastEnvironmentChange();
        }
    }

    public static void Remove(string ownedPath)
    {
        using RegistryKey key = Registry.CurrentUser.CreateSubKey(EnvironmentKey, writable: true);
        string current = key.GetValue("Path", string.Empty, RegistryValueOptions.DoNotExpandEnvironmentNames)
            as string ?? string.Empty;
        string updated = UpdatePathValue(current, ownedPath, add: false);
        if (!string.Equals(current, updated, StringComparison.Ordinal))
        {
            key.SetValue("Path", updated, RegistryValueKind.ExpandString);
            BroadcastEnvironmentChange();
        }
    }

    internal static string UpdatePathValue(string current, string ownedPath, bool add)
    {
        string normalizedOwned = Path.TrimEndingDirectorySeparator(Path.GetFullPath(ownedPath));
        List<string> retained = [];
        foreach (string raw in current.Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            string value = raw.Trim();
            if (value.Length == 0)
            {
                continue;
            }

            string comparable;
            try
            {
                comparable = Path.TrimEndingDirectorySeparator(Path.GetFullPath(
                    Environment.ExpandEnvironmentVariables(value)));
            }
            catch (Exception exception) when (exception is ArgumentException or NotSupportedException)
            {
                comparable = value.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            }

            if (!string.Equals(comparable, normalizedOwned, StringComparison.OrdinalIgnoreCase)
                && !retained.Contains(value, StringComparer.OrdinalIgnoreCase))
            {
                retained.Add(value);
            }
        }

        if (add)
        {
            retained.Add(normalizedOwned);
        }

        return string.Join(';', retained);
    }

    private static void BroadcastEnvironmentChange()
    {
        _ = SendMessageTimeout(
            new IntPtr(0xffff),
            0x001A,
            IntPtr.Zero,
            "Environment",
            0x0002,
            5000,
            out _);
    }

    [LibraryImport("user32.dll", EntryPoint = "SendMessageTimeoutW", StringMarshalling = StringMarshalling.Utf16)]
    private static partial IntPtr SendMessageTimeout(
        IntPtr window,
        uint message,
        IntPtr wordParameter,
        string longParameter,
        uint flags,
        uint timeout,
        out IntPtr result);
}
