using System.Diagnostics;
using System.Text;

namespace SwarmBootstrap;

internal sealed record ProcessRequest(
    string Executable,
    IReadOnlyList<string> Arguments,
    TimeSpan Timeout,
    string? WorkingDirectory = null,
    IReadOnlyDictionary<string, string>? Environment = null,
    IReadOnlySet<int>? SensitiveArgumentIndexes = null,
    int MaximumRetainedCharacters = 1_048_576);

internal sealed record ProcessResult(
    int ExitCode,
    string StandardOutput,
    string StandardError,
    bool TimedOut,
    TimeSpan Duration)
{
    public bool Succeeded => !TimedOut && ExitCode == 0;
}

internal interface IBoundedProcessRunner
{
    Task<ProcessResult> RunAsync(ProcessRequest request, CancellationToken cancellationToken);
}

internal sealed class BoundedProcess(BootstrapLogger logger) : IBoundedProcessRunner
{
    public async Task<ProcessResult> RunAsync(
        ProcessRequest request,
        CancellationToken cancellationToken)
    {
        if (request.Timeout <= TimeSpan.Zero || request.Timeout > TimeSpan.FromHours(1))
        {
            throw new ArgumentException("process timeout must be in (0, 3600] seconds");
        }

        if (request.MaximumRetainedCharacters is < 4096 or > 4_194_304)
        {
            throw new ArgumentException("retained process output limit is outside the safe range");
        }

        ProcessStartInfo start = new()
        {
            FileName = request.Executable,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        foreach (string argument in request.Arguments)
        {
            start.ArgumentList.Add(argument);
        }

        if (request.WorkingDirectory is not null)
        {
            start.WorkingDirectory = request.WorkingDirectory;
        }

        if (request.Environment is not null)
        {
            foreach ((string key, string value) in request.Environment)
            {
                start.Environment[key] = value;
            }
        }

        string display = RenderCommand(request);
        logger.Info($"process start timeout={request.Timeout.TotalSeconds:F0}s command={display}");
        Stopwatch stopwatch = Stopwatch.StartNew();
        using Process process = new() { StartInfo = start };
        try
        {
            if (!process.Start())
            {
                throw new IOException($"process did not start: {request.Executable}");
            }
        }
        catch (System.ComponentModel.Win32Exception exception)
        {
            throw new DependencyInstallException(
                $"could not start required executable {Path.GetFileName(request.Executable)}",
                exception);
        }

        TailTextBuffer stdout = new(request.MaximumRetainedCharacters);
        TailTextBuffer stderr = new(request.MaximumRetainedCharacters);
        Task outputTask = PumpAsync(process.StandardOutput, stdout);
        Task errorTask = PumpAsync(process.StandardError, stderr);
        using CancellationTokenSource timeout = new(request.Timeout);
        using CancellationTokenSource linked = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken,
            timeout.Token);
        bool timedOut = false;
        try
        {
            await process.WaitForExitAsync(linked.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (timeout.IsCancellationRequested)
        {
            timedOut = true;
            try
            {
                process.Kill(entireProcessTree: true);
            }
            catch (InvalidOperationException)
            {
                // The process exited while the timeout was being handled.
            }

            using CancellationTokenSource killWait = new(TimeSpan.FromSeconds(10));
            try
            {
                await process.WaitForExitAsync(killWait.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                logger.Error($"process tree did not terminate promptly: {display}");
            }
        }

        await Task.WhenAll(outputTask, errorTask).ConfigureAwait(false);
        stopwatch.Stop();
        int exitCode = timedOut || !process.HasExited ? -1 : process.ExitCode;
        ProcessResult result = new(
            exitCode,
            stdout.ToString(),
            stderr.ToString(),
            timedOut,
            stopwatch.Elapsed);
        logger.Info(
            $"process end exit={result.ExitCode} timed_out={result.TimedOut} "
            + $"duration_ms={result.Duration.TotalMilliseconds:F0} command={display}");
        if (!string.IsNullOrWhiteSpace(result.StandardError))
        {
            logger.Diagnostic($"process stderr command={display} detail={result.StandardError}");
        }

        return result;
    }

    private static async Task PumpAsync(StreamReader reader, TailTextBuffer destination)
    {
        char[] buffer = new char[4096];
        while (true)
        {
            int count = await reader.ReadAsync(buffer.AsMemory()).ConfigureAwait(false);
            if (count == 0)
            {
                return;
            }

            destination.Append(buffer.AsSpan(0, count));
        }
    }

    internal static string RenderCommand(ProcessRequest request)
    {
        IReadOnlySet<int> sensitive = request.SensitiveArgumentIndexes ?? new HashSet<int>();
        IEnumerable<string> arguments = request.Arguments.Select(
            (argument, index) => sensitive.Contains(index) ? "<redacted>" : Quote(argument));
        return $"{Quote(request.Executable)} {string.Join(" ", arguments)}".TrimEnd();
    }

    private static string Quote(string value)
    {
        if (value.Length > 0 && value.All(character => !char.IsWhiteSpace(character) && character != '"'))
        {
            return value;
        }

        return $"\"{value.Replace("\\", "\\\\", StringComparison.Ordinal).Replace("\"", "\\\"", StringComparison.Ordinal)}\"";
    }
}

internal sealed class TailTextBuffer(int capacity)
{
    private readonly char[] _buffer = new char[capacity];
    private int _count;
    private int _next;
    private long _discarded;

    public void Append(ReadOnlySpan<char> value)
    {
        foreach (char character in value)
        {
            if (_count < _buffer.Length)
            {
                _buffer[_next] = character;
                _count++;
                _next = (_next + 1) % _buffer.Length;
            }
            else
            {
                _buffer[_next] = character;
                _next = (_next + 1) % _buffer.Length;
                _discarded++;
            }
        }
    }

    public override string ToString()
    {
        StringBuilder result = new();
        if (_discarded > 0)
        {
            result.Append($"<truncated {_discarded} earlier characters>\n");
        }

        int start = _count == _buffer.Length ? _next : 0;
        for (int index = 0; index < _count; index++)
        {
            result.Append(_buffer[(start + index) % _buffer.Length]);
        }

        return result.ToString();
    }
}
