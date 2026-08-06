using System.Text.Json;

namespace SwarmBootstrap;

internal sealed record ServiceSnapshot(IReadOnlyList<string> TaskNames)
{
    public bool HadService => TaskNames.Count > 0;
}

internal sealed class ServiceLifecycle(
    IBoundedProcessRunner processes,
    BootstrapLogger logger,
    TimeSpan operationTimeout)
{
    private const string OwnedTaskPrefix = @"\SwarmInference\swarm-inference-";

    public async Task<ServiceSnapshot> CaptureAsync(CancellationToken cancellationToken)
    {
        string executable = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.System),
            "schtasks.exe");
        ProcessResult result = await processes.RunAsync(
            new ProcessRequest(executable, ["/Query", "/FO", "CSV", "/NH"], operationTimeout),
            cancellationToken).ConfigureAwait(false);
        if (!result.Succeeded)
        {
            throw new ServiceLifecycleException(
                $"could not enumerate scheduled tasks: {Diagnostic(result)}");
        }

        string[] owned = result.StandardOutput.Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries)
            .Select(ParseFirstCsvField)
            .Where(name => name.StartsWith(OwnedTaskPrefix, StringComparison.OrdinalIgnoreCase))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Order(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        return new ServiceSnapshot(owned);
    }

    public async Task StopAsync(ServiceSnapshot snapshot, CancellationToken cancellationToken)
    {
        string executable = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.System),
            "schtasks.exe");
        foreach (string task in snapshot.TaskNames)
        {
            ProcessResult result = await processes.RunAsync(
                new ProcessRequest(executable, ["/End", "/TN", task], operationTimeout),
                cancellationToken).ConfigureAwait(false);
            logger.Info(
                result.Succeeded
                    ? $"stopped owned scheduled task {task}"
                    : $"owned scheduled task was not running or could not be ended: {task}");
        }
    }

    public async Task RemoveAsync(ServiceSnapshot snapshot, CancellationToken cancellationToken)
    {
        string executable = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.System),
            "schtasks.exe");
        foreach (string task in snapshot.TaskNames)
        {
            ProcessResult result = await processes.RunAsync(
                new ProcessRequest(executable, ["/Delete", "/TN", task, "/F"], operationTimeout),
                cancellationToken).ConfigureAwait(false);
            if (!result.Succeeded)
            {
                throw new ServiceLifecycleException(
                    $"could not remove owned scheduled task {task}: {Diagnostic(result)}");
            }
        }
    }

    public static bool HasPreservedMembership(string stateRoot)
    {
        string security = Path.Combine(stateRoot, "security");
        return File.Exists(Path.Combine(security, "cluster.json"))
            && File.Exists(Path.Combine(security, "memberships.json"))
            && File.Exists(Path.Combine(security, "node-configuration.json"));
    }

    public async Task RestoreAsync(
        string swarmExecutable,
        string stateRoot,
        CancellationToken cancellationToken)
    {
        if (!HasPreservedMembership(stateRoot))
        {
            logger.Info("no valid preserved cluster membership was found; service remains deferred");
            return;
        }

        ProcessResult install = await processes.RunAsync(
            new ProcessRequest(
                swarmExecutable,
                ["node", "install-service", "--state-root", stateRoot, "--yes", "--json"],
                operationTimeout),
            cancellationToken).ConfigureAwait(false);
        if (!install.Succeeded)
        {
            throw new ServiceLifecycleException(
                $"preserved cluster service restoration failed: {Diagnostic(install)}");
        }

        DateTime deadline = DateTime.UtcNow.AddSeconds(Math.Min(90, operationTimeout.TotalSeconds));
        while (DateTime.UtcNow < deadline)
        {
            ProcessResult status = await processes.RunAsync(
                new ProcessRequest(
                    swarmExecutable,
                    ["node", "status", "--state-root", stateRoot, "--json"],
                    TimeSpan.FromSeconds(Math.Min(30, operationTimeout.TotalSeconds))),
                cancellationToken).ConfigureAwait(false);
            if (status.Succeeded && TryRuntimeState(status.StandardOutput, out string? state))
            {
                if (state == "ready")
                {
                    logger.Info("preserved cluster service reported a fresh ready state");
                    return;
                }

                if (state is "blocked" or "failed")
                {
                    throw new ServiceLifecycleException(
                        $"restored cluster service reported explicit {state} status");
                }
            }

            await Task.Delay(TimeSpan.FromSeconds(1), cancellationToken).ConfigureAwait(false);
        }

        throw new ServiceLifecycleException("restored cluster service did not report ready in time");
    }

    internal static string ParseFirstCsvField(string line)
    {
        if (line.Length == 0)
        {
            return string.Empty;
        }

        if (line[0] != '"')
        {
            int comma = line.IndexOf(',');
            return comma < 0 ? line.Trim() : line[..comma].Trim();
        }

        System.Text.StringBuilder result = new();
        for (int index = 1; index < line.Length; index++)
        {
            if (line[index] == '"')
            {
                if (index + 1 < line.Length && line[index + 1] == '"')
                {
                    result.Append('"');
                    index++;
                    continue;
                }

                return result.ToString();
            }

            result.Append(line[index]);
        }

        return result.ToString();
    }

    private static bool TryRuntimeState(string output, out string? state)
    {
        state = null;
        try
        {
            using JsonDocument document = JsonDocument.Parse(output);
            return FindState(document.RootElement, out state);
        }
        catch (JsonException)
        {
            return false;
        }
    }

    private static bool FindState(JsonElement element, out string? state)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                if (property.NameEquals("state") && property.Value.ValueKind == JsonValueKind.String)
                {
                    state = property.Value.GetString();
                    if (state is "ready" or "blocked" or "failed")
                    {
                        return true;
                    }
                }

                if (FindState(property.Value, out state))
                {
                    return true;
                }
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in element.EnumerateArray())
            {
                if (FindState(item, out state))
                {
                    return true;
                }
            }
        }

        state = null;
        return false;
    }

    private static string Diagnostic(ProcessResult result)
    {
        string value = string.IsNullOrWhiteSpace(result.StandardError)
            ? result.StandardOutput
            : result.StandardError;
        string redacted = BootstrapLogger.Redact(value.Trim());
        return redacted[..Math.Min(1000, redacted.Length)];
    }
}
