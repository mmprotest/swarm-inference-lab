using System.Text.Json;
using System.Text.Json.Serialization;

namespace SwarmBootstrap;

internal sealed record InstallRecord
{
    [JsonPropertyName("schema_version")]
    public int SchemaVersion { get; init; } = 1;

    [JsonPropertyName("product_version")]
    public required string ProductVersion { get; init; }

    [JsonPropertyName("git_tag")]
    public required string GitTag { get; init; }

    [JsonPropertyName("git_commit")]
    public required string GitCommit { get; init; }

    [JsonPropertyName("installation_mode")]
    public string InstallationMode { get; init; } = "native-windows";

    [JsonPropertyName("installation_operation")]
    public required string InstallationOperation { get; init; }

    [JsonPropertyName("candidate_backend")]
    public required string CandidateBackend { get; init; }

    [JsonPropertyName("selected_backend")]
    public required string SelectedBackend { get; init; }

    [JsonPropertyName("rejected_backend")]
    public string? RejectedBackend { get; init; }

    [JsonPropertyName("rejection_reason")]
    public string? RejectionReason { get; init; }

    [JsonPropertyName("python_version")]
    public required string PythonVersion { get; init; }

    [JsonPropertyName("uv_version")]
    public required string UvVersion { get; init; }

    [JsonPropertyName("uv_sha256")]
    public required string UvSha256 { get; init; }

    [JsonPropertyName("wheel_filename")]
    public required string WheelFilename { get; init; }

    [JsonPropertyName("wheel_sha256")]
    public required string WheelSha256 { get; init; }

    [JsonPropertyName("runtime_profile_filename")]
    public required string RuntimeProfileFilename { get; init; }

    [JsonPropertyName("runtime_profile_sha256")]
    public required string RuntimeProfileSha256 { get; init; }

    [JsonPropertyName("installed_at_utc")]
    public required string InstalledAtUtc { get; init; }

    [JsonPropertyName("installer_version")]
    public required string InstallerVersion { get; init; }

    [JsonPropertyName("signature_status")]
    public required string SignatureStatus { get; init; }

    [JsonPropertyName("previous_installed_version")]
    public string? PreviousInstalledVersion { get; init; }

    [JsonPropertyName("doctor_summary")]
    public required JsonElement DoctorSummary { get; init; }

    [JsonPropertyName("application_path")]
    public required string ApplicationPath { get; init; }

    [JsonPropertyName("state_path")]
    public required string StatePath { get; init; }

    [JsonPropertyName("release_manifest_sha256")]
    public required string ReleaseManifestSha256 { get; init; }

    public static InstallRecord? Load(PathInfo layout)
    {
        if (!File.Exists(layout.InstallRecord))
        {
            return null;
        }

        try
        {
            InstallRecord? value = JsonSerializer.Deserialize<InstallRecord>(
                File.ReadAllText(layout.InstallRecord),
                JsonDefaults.Strict);
            if (value is null || value.SchemaVersion != 1 || value.InstallationMode != "native-windows")
            {
                throw new ManifestException("installed record has an unsupported schema or mode");
            }

            return value;
        }
        catch (JsonException exception)
        {
            throw new ManifestException("installed record is malformed", exception);
        }
    }

    public void SaveAtomic(PathInfo layout)
    {
        Directory.CreateDirectory(layout.App);
        AtomicFile.WriteAllText(layout.InstallRecord, JsonSerializer.Serialize(this, JsonDefaults.Strict) + "\n");
    }
}

internal static class AtomicFile
{
    public static void WriteAllText(string path, string content)
    {
        string directory = Path.GetDirectoryName(path)
            ?? throw new IOException($"atomic destination has no parent: {path}");
        Directory.CreateDirectory(directory);
        string temporary = Path.Combine(directory, $".{Path.GetFileName(path)}.{Guid.NewGuid():N}.tmp");
        try
        {
            using (FileStream stream = new(
                temporary,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                4096,
                FileOptions.WriteThrough))
            using (StreamWriter writer = new(stream, new System.Text.UTF8Encoding(false)))
            {
                writer.Write(content);
                writer.Flush();
                stream.Flush(true);
            }

            File.Move(temporary, path, true);
        }
        finally
        {
            File.Delete(temporary);
        }
    }

    public static void WriteAllBytes(string path, byte[] content)
    {
        string directory = Path.GetDirectoryName(path)
            ?? throw new IOException($"atomic destination has no parent: {path}");
        Directory.CreateDirectory(directory);
        string temporary = Path.Combine(directory, $".{Path.GetFileName(path)}.{Guid.NewGuid():N}.tmp");
        try
        {
            using (FileStream stream = new(
                       temporary,
                       FileMode.CreateNew,
                       FileAccess.Write,
                       FileShare.None,
                       4096,
                       FileOptions.WriteThrough))
            {
                stream.Write(content);
                stream.Flush(true);
            }

            File.Move(temporary, path, true);
        }
        finally
        {
            File.Delete(temporary);
        }
    }
}
