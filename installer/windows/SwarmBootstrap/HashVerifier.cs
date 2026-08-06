using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace SwarmBootstrap;

internal static partial class HashVerifier
{
    private const string ZeroSha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000";

    [GeneratedRegex("^sha256:[0-9a-f]{64}$", RegexOptions.CultureInvariant)]
    private static partial Regex Sha256Pattern();

    [GeneratedRegex("^[0-9a-f]{40}$", RegexOptions.CultureInvariant)]
    private static partial Regex CommitPattern();

    public static string ComputeSha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return $"sha256:{Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant()}";
    }

    public static string ComputeSha256(ReadOnlySpan<byte> content) =>
        $"sha256:{Convert.ToHexString(SHA256.HashData(content)).ToLowerInvariant()}";

    public static void Verify(string path, FileAsset asset)
    {
        if (!File.Exists(path))
        {
            throw new ManifestException($"manifest file is missing: {asset.Filename}");
        }

        FileInfo info = new(path);
        if (info.Length != asset.SizeBytes)
        {
            throw new HashMismatchException($"size mismatch for {asset.Filename}");
        }

        string actual = ComputeSha256(path);
        if (!CryptographicOperations.FixedTimeEquals(
                System.Text.Encoding.ASCII.GetBytes(actual),
                System.Text.Encoding.ASCII.GetBytes(asset.Sha256)))
        {
            throw new HashMismatchException(
                $"SHA-256 mismatch for {asset.Filename}: expected {asset.Sha256}, got {actual}");
        }
    }

    public static ReleaseManifest LoadAndVerifyPayload(string payloadDirectory)
    {
        string manifestPath = Path.Combine(payloadDirectory, "release-manifest.json");
        if (!File.Exists(manifestPath))
        {
            throw new ManifestException("release-manifest.json is missing from the installer payload");
        }

        ReleaseManifest manifest;
        try
        {
            manifest = JsonSerializer.Deserialize<ReleaseManifest>(
                    File.ReadAllText(manifestPath),
                    JsonDefaults.Strict)
                ?? throw new ManifestException("release manifest is empty");
        }
        catch (JsonException exception)
        {
            throw new ManifestException("release manifest is not strict valid JSON", exception);
        }

        ValidateManifest(manifest);
        List<FileAsset> files =
        [
            manifest.Uv,
            manifest.Wheel,
            manifest.RuntimeProfiles.Cpu,
            manifest.RuntimeProfiles.Cuda,
            manifest.Bootstrapper,
            .. manifest.Payload,
        ];
        HashSet<string> expected = new(StringComparer.OrdinalIgnoreCase)
        {
            "release-manifest.json",
        };
        foreach (FileAsset asset in files)
        {
            if (!expected.Add(asset.Filename))
            {
                throw new ManifestException($"duplicate payload filename: {asset.Filename}");
            }

            Verify(Path.Combine(payloadDirectory, asset.Filename), asset);
        }

        string[] unexpected = Directory.EnumerateFiles(payloadDirectory)
            .Select(Path.GetFileName)
            .Where(name => name is not null && !expected.Contains(name))
            .Cast<string>()
            .Order(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (unexpected.Length > 0)
        {
            throw new ManifestException(
                $"installer payload contains unexpected files: {string.Join(", ", unexpected)}");
        }

        return manifest;
    }

    public static void ValidateManifest(ReleaseManifest manifest)
    {
        ValidateManifestScope(manifest, "embedded-payload");
    }

    public static ReleaseManifest ValidateReleaseManifest(
        byte[] content,
        ReleaseManifest embedded,
        string setupPath)
    {
        ReleaseManifest release;
        try
        {
            release = JsonSerializer.Deserialize<ReleaseManifest>(content, JsonDefaults.Strict)
                ?? throw new ManifestException("external release manifest is empty");
        }
        catch (JsonException exception)
        {
            throw new ManifestException(
                "external release manifest is not strict valid JSON",
                exception);
        }

        ValidateManifestScope(release, "release");
        ValidateMatchingReleaseIdentity(embedded, release);
        FileInfo setup = new(setupPath);
        if (!setup.Exists
            || release.Installer.Filename != setup.Name
            || release.Installer.SizeBytes != setup.Length)
        {
            throw new ManifestException(
                "external release manifest does not identify the executing setup file");
        }

        string actual = ComputeSha256(setup.FullName);
        if (!CryptographicOperations.FixedTimeEquals(
                System.Text.Encoding.ASCII.GetBytes(actual),
                System.Text.Encoding.ASCII.GetBytes(release.Installer.Sha256)))
        {
            throw new HashMismatchException(
                "executing setup SHA-256 does not match the external release manifest");
        }

        return release;
    }

    private static void ValidateManifestScope(ReleaseManifest manifest, string expectedScope)
    {
        if (manifest.SchemaVersion != 1
            || manifest.ManifestScope != expectedScope
            || manifest.Product != "swarm-inference-lab"
            || manifest.Architecture != "x86_64"
            || manifest.MinimumWindows != "10.0.22621")
        {
            throw new ManifestException("release manifest identity or target platform is invalid");
        }

        if (!CommitPattern().IsMatch(manifest.GitCommit))
        {
            throw new ManifestException("release manifest Git commit is invalid");
        }

        string expectedTag = VersionPolicy.ToGitTag(manifest.Version);
        if (!string.Equals(expectedTag, manifest.GitTag, StringComparison.Ordinal))
        {
            throw new ManifestException("release manifest version and tag do not agree");
        }

        bool prerelease = manifest.Version.Contains("rc", StringComparison.Ordinal);
        if (manifest.Channel != (prerelease ? "prerelease" : "stable"))
        {
            throw new ManifestException("release manifest version and channel do not agree");
        }

        if (!Version.TryParse(manifest.MinimumWindows, out _)
            || !System.Text.RegularExpressions.Regex.IsMatch(
                manifest.Python.Version,
                @"^3\.11\.\d+$",
                RegexOptions.CultureInvariant))
        {
            throw new ManifestException("release manifest runtime version pin is invalid");
        }

        foreach (FileAsset asset in EnumerateAssets(manifest))
        {
            if (Path.GetFileName(asset.Filename) != asset.Filename
                || asset.Filename.Contains('/')
                || asset.Filename.Contains('\\')
                || !Sha256Pattern().IsMatch(asset.Sha256)
                || asset.SizeBytes < 0)
            {
                throw new ManifestException($"invalid file identity for {asset.Filename}");
            }
        }

        ValidateSignature(manifest.Bootstrapper, prerelease, allowPlaceholder: false);
        bool embedded = expectedScope == "embedded-payload";
        ValidateSignature(manifest.Installer, prerelease, allowPlaceholder: embedded);
        if (embedded
            && (manifest.Installer.Sha256 != ZeroSha256 || manifest.Installer.SizeBytes != 0))
        {
            throw new ManifestException(
                "embedded payload manifest must use the documented installer self-hash placeholder");
        }

        if (!embedded
            && (manifest.Installer.Sha256 == ZeroSha256 || manifest.Installer.SizeBytes <= 0))
        {
            throw new ManifestException(
                "external release manifest must identify the final setup executable");
        }

        if (!prerelease
            && (manifest.Bootstrapper.SignatureStatus != "signed"
                || manifest.Installer.SignatureStatus != "signed"))
        {
            throw new ManifestException("stable payloads require signed executables");
        }
    }

    private static void ValidateMatchingReleaseIdentity(
        ReleaseManifest embedded,
        ReleaseManifest release)
    {
        bool identityMatches = embedded.SchemaVersion == release.SchemaVersion
            && embedded.Product == release.Product
            && embedded.Version == release.Version
            && embedded.GitTag == release.GitTag
            && embedded.GitCommit == release.GitCommit
            && embedded.Channel == release.Channel
            && embedded.BuiltAtUtc == release.BuiltAtUtc
            && embedded.MinimumWindows == release.MinimumWindows
            && embedded.Architecture == release.Architecture
            && embedded.Python.Version == release.Python.Version
            && SameUv(embedded.Uv, release.Uv)
            && SameAsset(embedded.Wheel, release.Wheel)
            && SameAsset(embedded.RuntimeProfiles.Cpu, release.RuntimeProfiles.Cpu)
            && SameAsset(embedded.RuntimeProfiles.Cuda, release.RuntimeProfiles.Cuda)
            && SameSignedAsset(embedded.Bootstrapper, release.Bootstrapper)
            && embedded.Installer.SignatureStatus == release.Installer.SignatureStatus
            && embedded.Installer.SignatureVerification == release.Installer.SignatureVerification
            && embedded.Installer.PublisherSubject == release.Installer.PublisherSubject
            && embedded.Payload.Count == release.Payload.Count
            && embedded.Payload.Zip(release.Payload, SameAsset).All(matches => matches);
        if (!identityMatches)
        {
            throw new ManifestException(
                "external release manifest does not match the embedded payload identity");
        }
    }

    private static bool SameAsset(FileAsset left, FileAsset right) =>
        left.Filename == right.Filename
        && left.Sha256 == right.Sha256
        && left.SizeBytes == right.SizeBytes;

    private static bool SameUv(UvAsset left, UvAsset right) =>
        left.Version == right.Version && SameAsset(left, right);

    private static bool SameSignedAsset(SignedAsset left, SignedAsset right) =>
        SameAsset(left, right)
        && left.SignatureStatus == right.SignatureStatus
        && left.SignatureVerification == right.SignatureVerification
        && left.PublisherSubject == right.PublisherSubject;

    private static IEnumerable<FileAsset> EnumerateAssets(ReleaseManifest manifest)
    {
        yield return manifest.Uv;
        yield return manifest.Wheel;
        yield return manifest.RuntimeProfiles.Cpu;
        yield return manifest.RuntimeProfiles.Cuda;
        yield return manifest.Bootstrapper;
        yield return manifest.Installer;
        foreach (FileAsset asset in manifest.Payload)
        {
            yield return asset;
        }
    }

    private static void ValidateSignature(SignedAsset asset, bool prerelease, bool allowPlaceholder)
    {
        if (asset.SignatureStatus == "signed")
        {
            if (asset.SignatureVerification != "valid" || string.IsNullOrWhiteSpace(asset.PublisherSubject))
            {
                throw new ManifestException($"signed file {asset.Filename} lacks verification metadata");
            }
        }
        else if (asset.SignatureStatus == "unsigned-prerelease")
        {
            if (!prerelease
                || asset.SignatureVerification != "not-signed"
                || asset.PublisherSubject is not null)
            {
                throw new ManifestException($"unsigned status for {asset.Filename} is not valid");
            }
        }
        else
        {
            throw new ManifestException($"unknown signature status for {asset.Filename}");
        }

        if (!allowPlaceholder && asset.Sha256 == ZeroSha256)
        {
            throw new ManifestException($"{asset.Filename} cannot use a placeholder hash");
        }
    }
}

internal static partial class VersionPolicy
{
    [GeneratedRegex(@"^(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?$", RegexOptions.CultureInvariant)]
    private static partial Regex Pep440Pattern();

    public static string ToGitTag(string version)
    {
        Match match = Pep440Pattern().Match(version);
        if (!match.Success)
        {
            throw new ManifestException($"unsupported package version: {version}");
        }

        string release = $"{match.Groups[1].Value}.{match.Groups[2].Value}.{match.Groups[3].Value}";
        return match.Groups[4].Success ? $"v{release}-rc.{match.Groups[4].Value}" : $"v{release}";
    }

    public static int Compare(string left, string right)
    {
        Match leftMatch = Pep440Pattern().Match(left);
        Match rightMatch = Pep440Pattern().Match(right);
        if (!leftMatch.Success || !rightMatch.Success)
        {
            throw new ManifestException("installed or candidate version is not supported PEP 440") ;
        }

        for (int index = 1; index <= 3; index++)
        {
            int comparison = int.Parse(leftMatch.Groups[index].Value, System.Globalization.CultureInfo.InvariantCulture)
                .CompareTo(int.Parse(rightMatch.Groups[index].Value, System.Globalization.CultureInfo.InvariantCulture));
            if (comparison != 0)
            {
                return comparison;
            }
        }

        bool leftRc = leftMatch.Groups[4].Success;
        bool rightRc = rightMatch.Groups[4].Success;
        if (leftRc != rightRc)
        {
            return leftRc ? -1 : 1;
        }

        return leftRc
            ? int.Parse(leftMatch.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture)
                .CompareTo(int.Parse(rightMatch.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture))
            : 0;
    }
}
