using System.Text.Json;
using System.Text.Json.Serialization;

namespace SwarmBootstrap;

internal static class JsonDefaults
{
    public static readonly JsonSerializerOptions Strict = new()
    {
        PropertyNameCaseInsensitive = false,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
        WriteIndented = true,
    };
}

internal sealed record ReleaseManifest
{
    [JsonPropertyName("schema_version")]
    public required int SchemaVersion { get; init; }

    [JsonPropertyName("manifest_scope")]
    public required string ManifestScope { get; init; }

    [JsonPropertyName("product")]
    public required string Product { get; init; }

    [JsonPropertyName("version")]
    public required string Version { get; init; }

    [JsonPropertyName("git_tag")]
    public required string GitTag { get; init; }

    [JsonPropertyName("git_commit")]
    public required string GitCommit { get; init; }

    [JsonPropertyName("channel")]
    public required string Channel { get; init; }

    [JsonPropertyName("built_at_utc")]
    public required string BuiltAtUtc { get; init; }

    [JsonPropertyName("minimum_windows")]
    public required string MinimumWindows { get; init; }

    [JsonPropertyName("architecture")]
    public required string Architecture { get; init; }

    [JsonPropertyName("python")]
    public required PythonAsset Python { get; init; }

    [JsonPropertyName("uv")]
    public required UvAsset Uv { get; init; }

    [JsonPropertyName("wheel")]
    public required FileAsset Wheel { get; init; }

    [JsonPropertyName("runtime_profiles")]
    public required RuntimeProfiles RuntimeProfiles { get; init; }

    [JsonPropertyName("bootstrapper")]
    public required SignedAsset Bootstrapper { get; init; }

    [JsonPropertyName("installer")]
    public required SignedAsset Installer { get; init; }

    [JsonPropertyName("payload")]
    public required IReadOnlyList<FileAsset> Payload { get; init; }
}

internal sealed record PythonAsset
{
    [JsonPropertyName("version")]
    public required string Version { get; init; }
}

internal record FileAsset
{
    [JsonPropertyName("filename")]
    public required string Filename { get; init; }

    [JsonPropertyName("sha256")]
    public required string Sha256 { get; init; }

    [JsonPropertyName("size_bytes")]
    public required long SizeBytes { get; init; }
}

internal sealed record UvAsset : FileAsset
{
    [JsonPropertyName("version")]
    public required string Version { get; init; }
}

internal sealed record SignedAsset : FileAsset
{
    [JsonPropertyName("signature_status")]
    public required string SignatureStatus { get; init; }

    [JsonPropertyName("publisher_subject")]
    public string? PublisherSubject { get; init; }

    [JsonPropertyName("signature_verification")]
    public required string SignatureVerification { get; init; }
}

internal sealed record RuntimeProfiles
{
    [JsonPropertyName("windows-x64-cpu")]
    public required FileAsset Cpu { get; init; }

    [JsonPropertyName("windows-x64-cuda")]
    public required FileAsset Cuda { get; init; }
}

internal enum BackendProfile
{
    Auto,
    Cpu,
    Cuda,
}

internal sealed record CandidateRuntime(
    BackendProfile Backend,
    string RuntimePath,
    JsonElement Doctor,
    string SelectedBackend,
    string ProfileFilename,
    string ProfileSha256);

internal sealed record InstallationResult(
    string Operation,
    string Version,
    string GitTag,
    string SelectedProfile,
    string? RejectedProfile,
    string? RejectionReason,
    string InstallRoot,
    string StateRoot,
    string LogPath,
    JsonElement Doctor);
