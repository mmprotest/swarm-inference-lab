namespace SwarmBootstrap;

internal sealed class RuntimeTransaction : IDisposable
{
    private readonly PathInfo _layout;
    private readonly BootstrapLogger _logger;
    private readonly bool _keepFailedStaging;
    private readonly byte[]? _previousRecord;
    private readonly byte[]? _previousManifest;
    private string? _previousRuntime;
    private string? _stagingRuntime;
    private string? _previousPayloadCache;
    private string? _stagingPayloadCache;
    private bool _activeMoved;
    private bool _published;
    private bool _payloadPublished;
    private bool _committed;
    private bool _rolledBack;

    public RuntimeTransaction(PathInfo layout, BootstrapLogger logger, bool keepFailedStaging)
    {
        _layout = layout;
        _logger = logger;
        _keepFailedStaging = keepFailedStaging;
        _previousRecord = File.Exists(layout.InstallRecord)
            ? File.ReadAllBytes(layout.InstallRecord)
            : null;
        _previousManifest = File.Exists(layout.Manifest) ? File.ReadAllBytes(layout.Manifest) : null;
    }

    public string CreateStagingRuntime()
    {
        if (_stagingRuntime is not null)
        {
            SafeDeleteDirectory(_stagingRuntime);
        }

        _stagingRuntime = _layout.NewStagingRuntime();
        Directory.CreateDirectory(_stagingRuntime);
        return _stagingRuntime;
    }

    public void DiscardStaging()
    {
        if (_stagingRuntime is not null && !_keepFailedStaging)
        {
            SafeDeleteDirectory(_stagingRuntime);
        }

        _stagingRuntime = null;
    }

    public string CreateStagingPayloadCache()
    {
        if (_stagingPayloadCache is not null)
        {
            SafeDeleteDirectory(_stagingPayloadCache);
        }

        _stagingPayloadCache = Path.Combine(
            _layout.Root,
            $".payload-cache.{Guid.NewGuid():N}.staging");
        Directory.CreateDirectory(_stagingPayloadCache);
        return _stagingPayloadCache;
    }

    public void MoveActiveAside(InstallRecord? previous)
    {
        if (!Directory.Exists(_layout.Runtime))
        {
            return;
        }

        string label = previous?.ProductVersion ?? "unknown";
        string safeLabel = string.Concat(label.Select(character => char.IsLetterOrDigit(character) ? character : '-'));
        _previousRuntime = Path.Combine(
            _layout.Previous,
            $"runtime-{safeLabel}-{DateTime.UtcNow:yyyyMMddHHmmss}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_layout.Previous);
        Directory.Move(_layout.Runtime, _previousRuntime);
        _activeMoved = true;
        _logger.Info($"active runtime moved to rollback slot {_previousRuntime}");
    }

    public void PublishStaging()
    {
        if (_stagingRuntime is null || !Directory.Exists(_stagingRuntime))
        {
            throw new IOException("candidate staging runtime does not exist");
        }

        if (Directory.Exists(_layout.Runtime))
        {
            throw new IOException("stable runtime path is unexpectedly occupied");
        }

        Directory.Move(_stagingRuntime, _layout.Runtime);
        _stagingRuntime = null;
        _published = true;
        _logger.Info("candidate runtime published at the stable runtime path");
    }

    public void PublishPayloadCache()
    {
        if (_stagingPayloadCache is null || !Directory.Exists(_stagingPayloadCache))
        {
            throw new IOException("candidate payload cache does not exist");
        }

        if (Directory.Exists(_layout.PayloadCache))
        {
            _previousPayloadCache = Path.Combine(
                _layout.Previous,
                $"payload-cache-{DateTime.UtcNow:yyyyMMddHHmmss}-{Guid.NewGuid():N}");
            Directory.CreateDirectory(_layout.Previous);
            Directory.Move(_layout.PayloadCache, _previousPayloadCache);
        }

        Directory.Move(_stagingPayloadCache, _layout.PayloadCache);
        _stagingPayloadCache = null;
        _payloadPublished = true;
        _logger.Info("candidate payload cache published transactionally");
    }

    public void Commit()
    {
        _committed = true;
        if (_previousRuntime is not null)
        {
            TryDeleteCommittedDirectory(_previousRuntime, "previous runtime");
            _previousRuntime = null;
        }

        if (_previousPayloadCache is not null)
        {
            TryDeleteCommittedDirectory(_previousPayloadCache, "previous payload cache");
            _previousPayloadCache = null;
        }

        _logger.Info("runtime transaction committed; previous runtime removed");
    }

    public void Rollback()
    {
        if (_committed || _rolledBack)
        {
            return;
        }

        if (_published && Directory.Exists(_layout.Runtime))
        {
            SafeDeleteDirectory(_layout.Runtime);
        }
        _published = false;

        if (_activeMoved && _previousRuntime is not null && Directory.Exists(_previousRuntime))
        {
            Directory.Move(_previousRuntime, _layout.Runtime);
            _previousRuntime = null;
            _activeMoved = false;
        }

        if (_payloadPublished && Directory.Exists(_layout.PayloadCache))
        {
            SafeDeleteDirectory(_layout.PayloadCache);
        }
        _payloadPublished = false;

        if (_previousPayloadCache is not null && Directory.Exists(_previousPayloadCache))
        {
            Directory.Move(_previousPayloadCache, _layout.PayloadCache);
            _previousPayloadCache = null;
        }

        RestoreFile(_layout.InstallRecord, _previousRecord);
        RestoreFile(_layout.Manifest, _previousManifest);
        if (_stagingRuntime is not null && !_keepFailedStaging)
        {
            SafeDeleteDirectory(_stagingRuntime);
            _stagingRuntime = null;
        }

        if (_stagingPayloadCache is not null && !_keepFailedStaging)
        {
            SafeDeleteDirectory(_stagingPayloadCache);
            _stagingPayloadCache = null;
        }

        _rolledBack = true;
        _logger.Info("runtime transaction rolled back to the previous installation");
    }

    public void Dispose()
    {
        if (!_committed)
        {
            Rollback();
        }
    }

    private void SafeDeleteDirectory(string path)
    {
        if (!PathInfo.IsSameOrChild(path, _layout.Root)
            || string.Equals(
                Path.TrimEndingDirectorySeparator(path),
                Path.TrimEndingDirectorySeparator(_layout.Root),
                StringComparison.OrdinalIgnoreCase))
        {
            throw new IOException($"refusing to remove an unowned directory: {path}");
        }

        if (Directory.Exists(path))
        {
            RuntimeInstaller.DeleteOwnedDirectory(path, _layout.Root);
        }
    }

    private void TryDeleteCommittedDirectory(string path, string label)
    {
        Exception? lastFailure = null;
        for (int attempt = 1; attempt <= 6; attempt++)
        {
            try
            {
                SafeDeleteDirectory(path);
                return;
            }
            catch (Exception exception) when (
                exception is IOException or UnauthorizedAccessException)
            {
                lastFailure = exception;
                if (attempt < 6)
                {
                    Thread.Sleep(TimeSpan.FromMilliseconds(200 * attempt));
                }
            }
        }

        _logger.Error($"{label} cleanup was deferred: {lastFailure!.Message}");
    }

    private static void RestoreFile(string path, byte[]? content)
    {
        if (content is null)
        {
            File.Delete(path);
        }
        else
        {
            AtomicFile.WriteAllBytes(path, content);
        }
    }
}
