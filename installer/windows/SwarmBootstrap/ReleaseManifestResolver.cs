using System.Net;

namespace SwarmBootstrap;

internal sealed record ReleaseManifestEvidence(
    byte[] Content,
    string Sha256,
    string Source,
    ReleaseManifest Manifest);

internal static class ReleaseManifestResolver
{
    private const int MaximumManifestBytes = 8 * 1024 * 1024;
    private static readonly HashSet<string> AllowedHosts = new(StringComparer.OrdinalIgnoreCase)
    {
        "github.com",
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
    };

    public static async Task<ReleaseManifestEvidence> ResolveAsync(
        ReleaseManifest embedded,
        string payloadDirectory,
        string? setupPath,
        BootstrapLogger logger,
        CancellationToken cancellationToken)
    {
        if (setupPath is null)
        {
            byte[] embeddedContent = await ReadBoundedFileAsync(
                    Path.Combine(payloadDirectory, "release-manifest.json"),
                    cancellationToken)
                .ConfigureAwait(false);
            logger.Info(
                "no --setup-path was supplied; installation evidence uses the verified embedded manifest");
            return new ReleaseManifestEvidence(
                embeddedContent,
                HashVerifier.ComputeSha256(embeddedContent),
                "embedded-payload",
                embedded);
        }

        string setup = Path.GetFullPath(setupPath);
        if (!File.Exists(setup)
            || !string.Equals(
                Path.GetFileName(setup),
                "SwarmInferenceSetup-x64.exe",
                StringComparison.OrdinalIgnoreCase))
        {
            throw new ManifestException("--setup-path must identify SwarmInferenceSetup-x64.exe");
        }

        string sidecar = Path.Combine(
            Path.GetDirectoryName(setup)
                ?? throw new ManifestException("setup path has no parent directory"),
            "release-manifest.json");
        byte[] content;
        string source;
        if (File.Exists(sidecar))
        {
            content = await ReadBoundedFileAsync(sidecar, cancellationToken).ConfigureAwait(false);
            source = "release-sidecar";
        }
        else
        {
            try
            {
                content = await DownloadAsync(embedded.GitTag, cancellationToken).ConfigureAwait(false);
            }
            catch (HttpRequestException exception)
            {
                throw new ManifestException(
                    "bounded GitHub release manifest request failed",
                    exception);
            }

            source = "github-release";
        }

        ReleaseManifest release = HashVerifier.ValidateReleaseManifest(content, embedded, setup);
        string digest = HashVerifier.ComputeSha256(content);
        logger.Info($"external release manifest verified source={source} sha256={digest}");
        return new ReleaseManifestEvidence(content, digest, source, release);
    }

    private static async Task<byte[]> ReadBoundedFileAsync(
        string path,
        CancellationToken cancellationToken)
    {
        FileInfo info = new(path);
        if (!info.Exists || info.Length is <= 0 or > MaximumManifestBytes)
        {
            throw new ManifestException("release manifest sidecar has an invalid bounded size");
        }

        await using FileStream stream = new(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            65536,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        return await ReadBoundedAsync(stream, cancellationToken).ConfigureAwait(false);
    }

    private static async Task<byte[]> DownloadAsync(
        string tag,
        CancellationToken cancellationToken)
    {
        Uri current = new(
            $"https://github.com/mmprotest/swarm-inference-lab/releases/download/"
            + $"{Uri.EscapeDataString(tag)}/release-manifest.json");
        using HttpClientHandler handler = new()
        {
            AllowAutoRedirect = false,
            AutomaticDecompression = DecompressionMethods.None,
            UseCookies = false,
        };
        using HttpClient client = new(handler)
        {
            Timeout = TimeSpan.FromSeconds(60),
        };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("SwarmBootstrap/1");
        client.DefaultRequestHeaders.Accept.ParseAdd("application/json");
        for (int redirect = 0; redirect <= 5; redirect++)
        {
            ValidateUri(current);
            using HttpRequestMessage request = new(HttpMethod.Get, current);
            using HttpResponseMessage response = await client.SendAsync(
                    request,
                    HttpCompletionOption.ResponseHeadersRead,
                    cancellationToken)
                .ConfigureAwait(false);
            if (IsRedirect(response.StatusCode))
            {
                Uri? location = response.Headers.Location;
                if (location is null || redirect == 5)
                {
                    throw new ManifestException("release manifest download has an invalid redirect");
                }

                current = location.IsAbsoluteUri ? location : new Uri(current, location);
                continue;
            }

            if (response.StatusCode != HttpStatusCode.OK)
            {
                throw new ManifestException(
                    $"release manifest download failed with HTTP {(int)response.StatusCode}");
            }

            if (response.Content.Headers.ContentLength is long length
                && (length <= 0 || length > MaximumManifestBytes))
            {
                throw new ManifestException("release manifest response has an invalid bounded size");
            }

            await using Stream stream = await response.Content.ReadAsStreamAsync(cancellationToken)
                .ConfigureAwait(false);
            return await ReadBoundedAsync(stream, cancellationToken).ConfigureAwait(false);
        }

        throw new ManifestException("release manifest download exceeded its redirect limit");
    }

    private static async Task<byte[]> ReadBoundedAsync(
        Stream stream,
        CancellationToken cancellationToken)
    {
        using MemoryStream content = new();
        byte[] buffer = new byte[65536];
        while (true)
        {
            int count = await stream.ReadAsync(buffer, cancellationToken).ConfigureAwait(false);
            if (count == 0)
            {
                break;
            }

            if (content.Length + count > MaximumManifestBytes)
            {
                throw new ManifestException("release manifest exceeded its bounded size");
            }

            content.Write(buffer, 0, count);
        }

        if (content.Length == 0)
        {
            throw new ManifestException("release manifest is empty");
        }

        return content.ToArray();
    }

    private static bool IsRedirect(HttpStatusCode status) => status is
        HttpStatusCode.MovedPermanently
        or HttpStatusCode.Found
        or HttpStatusCode.SeeOther
        or HttpStatusCode.TemporaryRedirect
        or HttpStatusCode.PermanentRedirect;

    internal static void ValidateUri(Uri uri)
    {
        if (uri.Scheme != Uri.UriSchemeHttps || !AllowedHosts.Contains(uri.Host))
        {
            throw new ManifestException(
                $"release manifest download redirected to an untrusted host: {uri.Host}");
        }
    }
}
