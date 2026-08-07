#ifndef PayloadDir
  #error PayloadDir must point to a complete verified embedded payload
#endif
#ifndef OutputDir
  #error OutputDir must be defined
#endif
#ifndef ProductVersion
  #error ProductVersion must be defined
#endif
#ifndef DisplayVersion
  #error DisplayVersion must be defined
#endif
#ifndef WheelFilename
  #error WheelFilename must be defined
#endif
#ifndef BootstrapperSha256
  #error BootstrapperSha256 must be defined
#endif

#define ProductName "Swarm Inference"
#define ProductPublisher "swarm-inference-lab contributors"
#define ProductUrl "https://github.com/mmprotest/swarm-inference-lab"
#define ProductAppId "{{946CB20D-3399-4C3C-AD55-41C851C02E56}"

[Setup]
AppId={#ProductAppId}
AppName={#ProductName}
AppVersion={#ProductVersion}
AppVerName={#ProductName} {#ProductVersion}
AppPublisher={#ProductPublisher}
AppPublisherURL={#ProductUrl}
AppSupportURL={#ProductUrl}/issues
AppUpdatesURL={#ProductUrl}/releases
VersionInfoVersion={#DisplayVersion}
VersionInfoCompany={#ProductPublisher}
VersionInfoDescription=Native per-user installer for Swarm Inference
VersionInfoProductName={#ProductName}
VersionInfoProductVersion={#DisplayVersion}
DefaultDirName={localappdata}\Programs\SwarmInference
DefaultGroupName=Swarm Inference
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupArchitecture=x64
MinVersion=10.0.22621
; uv-managed Python maintains a version junction inside this non-elevated,
; current-user-only app root. RedirectionGuard blocks that legitimate junction.
RedirectionGuard=no
LicenseFile={#PayloadDir}\LICENSE
SetupIconFile=assets\swarm.ico
WizardSmallImageFile=assets\wizard-small.bmp
WizardImageFile=assets\wizard-large.bmp
OutputDir={#OutputDir}
OutputBaseFilename=SwarmInferenceSetup-x64
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
Uninstallable=yes
UninstallDisplayName={#ProductName} {#ProductVersion}
UninstallDisplayIcon={app}\app\swarm.ico
UsePreviousAppDir=yes
UsePreviousGroup=yes
ChangesEnvironment=yes
CloseApplications=yes
RestartApplications=no
RestartIfNeededByRun=no
AllowNoIcons=yes
WizardStyle=modern

[Files]
Source: "{#PayloadDir}\SwarmBootstrap.exe"; Flags: dontcopy
Source: "{#PayloadDir}\uv.exe"; Flags: dontcopy
Source: "{#PayloadDir}\{#WheelFilename}"; Flags: dontcopy
Source: "{#PayloadDir}\windows-x64-cpu.requirements.lock"; Flags: dontcopy
Source: "{#PayloadDir}\windows-x64-cuda.requirements.lock"; Flags: dontcopy
Source: "{#PayloadDir}\llama-b9637-bin-win-cpu-x64.zip"; Flags: dontcopy
Source: "{#PayloadDir}\llama-b9637-bin-win-cuda-13.3-x64.zip"; Flags: dontcopy
Source: "{#PayloadDir}\cudart-llama-bin-win-cuda-13.3-x64.zip"; Flags: dontcopy
Source: "{#PayloadDir}\release-manifest.json"; Flags: dontcopy
Source: "{#PayloadDir}\LICENSE"; Flags: dontcopy
Source: "{#PayloadDir}\swarm.ico"; Flags: dontcopy
Source: "{#PayloadDir}\wizard-small.bmp"; Flags: dontcopy
Source: "{#PayloadDir}\wizard-large.bmp"; Flags: dontcopy
Source: "{#PayloadDir}\SwarmBootstrap.exe"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "assets\swarm.ico"; DestDir: "{app}\app"; Flags: ignoreversion

[Icons]
Name: "{group}\Swarm Inference"; Filename: "{app}\runtime\Scripts\swarm.exe"; Parameters: "--help"; WorkingDir: "{localappdata}"; IconFilename: "{app}\app\swarm.ico"

[Registry]
Root: HKCU; Subkey: "Software\SwarmInference"; ValueType: string; ValueName: "PurgeStateOnUninstall"; ValueData: "{code:PurgePreference}"; Flags: uninsdeletevalue

[UninstallDelete]
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\update-staging"
Type: dirifempty; Name: "{app}\bin"
Type: dirifempty; Name: "{app}"

[Code]
var
  PurgeStateCheck: TNewCheckBox;
  InstalledBackend: String;

function QuoteArgument(const Value: String): String;
begin
  Result := Value;
  StringChangeEx(Result, '"', '\"', True);
  Result := '"' + Result + '"';
end;

function RequestedBackend: String;
begin
  Result := Lowercase(ExpandConstant('{param:BACKEND|auto}'));
  if (Result <> 'auto') and (Result <> 'cpu') and (Result <> 'cuda') then
    RaiseException('/BACKEND must be auto, cpu, or cuda');
end;

function DowngradeFlag: String;
var
  Parameter: String;
begin
  Parameter := Lowercase(ExpandConstant('{param:ALLOWDOWNGRADE|0}'));
  if (Parameter <> '0') and (Parameter <> '1') then
    RaiseException('/ALLOWDOWNGRADE must be 0 or 1');
  if Parameter = '1' then
    Result := ' --allow-downgrade'
  else
    Result := '';
end;

function BootstrapLogPath: String;
var
  SetupLog: String;
begin
  SetupLog := ExpandConstant('{param:LOG|}');
  if SetupLog <> '' then
    Result := SetupLog + '.bootstrapper.log'
  else
    Result := ExpandConstant('{app}\logs\bootstrapper-inno.log');
end;

function PurgeRequested: Boolean;
var
  Parameter: String;
begin
  Parameter := Lowercase(ExpandConstant('{param:PURGESTATE|}'));
  if Parameter <> '' then
  begin
    if (Parameter <> '0') and (Parameter <> '1') then
      RaiseException('/PURGESTATE must be 0 or 1');
    Result := Parameter = '1';
  end
  else
    Result := Assigned(PurgeStateCheck) and PurgeStateCheck.Checked;
end;

function PurgePreference(Param: String): String;
begin
  if PurgeRequested then
    Result := '1'
  else
    Result := '0';
end;

procedure CopyPayloadFile(const Filename, Destination: String);
var
  Source: String;
begin
  ExtractTemporaryFile(Filename);
  Source := ExpandConstant('{tmp}\') + Filename;
  if not FileCopy(Source, Destination, False) then
    RaiseException('Could not stage verified payload file ' + Filename);
end;

procedure StagePayload;
var
  Payload: String;
begin
  Payload := ExpandConstant('{tmp}\swarm-payload');
  if DirExists(Payload) then
    DelTree(Payload, True, True, True);
  ForceDirectories(Payload);
  CopyPayloadFile('SwarmBootstrap.exe', Payload + '\SwarmBootstrap.exe');
  CopyPayloadFile('uv.exe', Payload + '\uv.exe');
  CopyPayloadFile('{#WheelFilename}', Payload + '\{#WheelFilename}');
  CopyPayloadFile('windows-x64-cpu.requirements.lock', Payload + '\windows-x64-cpu.requirements.lock');
  CopyPayloadFile('windows-x64-cuda.requirements.lock', Payload + '\windows-x64-cuda.requirements.lock');
  CopyPayloadFile('llama-b9637-bin-win-cpu-x64.zip', Payload + '\llama-b9637-bin-win-cpu-x64.zip');
  CopyPayloadFile('llama-b9637-bin-win-cuda-13.3-x64.zip', Payload + '\llama-b9637-bin-win-cuda-13.3-x64.zip');
  CopyPayloadFile('cudart-llama-bin-win-cuda-13.3-x64.zip', Payload + '\cudart-llama-bin-win-cuda-13.3-x64.zip');
  CopyPayloadFile('release-manifest.json', Payload + '\release-manifest.json');
  CopyPayloadFile('LICENSE', Payload + '\LICENSE');
  CopyPayloadFile('swarm.ico', Payload + '\swarm.ico');
  CopyPayloadFile('wizard-small.bmp', Payload + '\wizard-small.bmp');
  CopyPayloadFile('wizard-large.bmp', Payload + '\wizard-large.bmp');
  if CompareText(GetSHA256OfFile(Payload + '\SwarmBootstrap.exe'), '{#BootstrapperSha256}') <> 0 then
    RaiseException('Extracted bootstrapper failed its pinned SHA-256 check');
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ExitCode: Integer;
  Payload: String;
  Parameters: String;
  LogPath: String;
begin
  Result := '';
  try
    StagePayload;
    Payload := ExpandConstant('{tmp}\swarm-payload');
    LogPath := BootstrapLogPath;
    WizardForm.StatusLabel.Caption := 'Installing the verified ' + RequestedBackend + ' runtime...';
    Parameters := 'install --payload ' + QuoteArgument(Payload) +
      ' --install-root ' + QuoteArgument(ExpandConstant('{app}')) +
      ' --setup-path ' + QuoteArgument(ExpandConstant('{srcexe}')) +
      ' --backend ' + RequestedBackend +
      ' --timeout-seconds 1800 --json --log ' + QuoteArgument(LogPath) + DowngradeFlag;
    if not Exec(Payload + '\SwarmBootstrap.exe', Parameters, '', SW_HIDE,
      ewWaitUntilTerminated, ExitCode) then
      Result := 'The native installation engine could not be started. Log: ' + LogPath
    else if ExitCode <> 0 then
      Result := 'The native installation engine failed with exit code ' + IntToStr(ExitCode) +
        '. The previous working runtime was restored when this was an upgrade.' + #13#10 +
        'Diagnostic log: ' + LogPath;
  except
    Result := GetExceptionMessage;
  end;
end;

function ReadJsonString(const Text, Name: String): String;
var
  Marker: String;
  StartPosition: Integer;
  EndPosition: Integer;
  Remaining: String;
begin
  Result := '';
  Marker := '"' + Name + '": "';
  StartPosition := Pos(Marker, Text);
  if StartPosition = 0 then
    Exit;
  StartPosition := StartPosition + Length(Marker);
  Remaining := Copy(Text, StartPosition, Length(Text) - StartPosition + 1);
  EndPosition := Pos('"', Remaining);
  if EndPosition > 0 then
    Result := Copy(Remaining, 1, EndPosition - 1);
end;

procedure CurPageChanged(CurPageID: Integer);
var
  RecordText: AnsiString;
begin
  if CurPageID = wpFinished then
  begin
    if LoadStringFromFile(ExpandConstant('{app}\app\install-record.json'), RecordText) then
      InstalledBackend := ReadJsonString(String(RecordText), 'selected_backend');
    if InstalledBackend = '' then
      InstalledBackend := 'verified';
    WizardForm.FinishedLabel.Caption :=
      'Swarm Inference was installed with the ' + InstalledBackend + ' profile.' + #13#10 + #13#10 +
      'Open a new terminal, then run:' + #13#10 +
      'swarm --version' + #13#10 +
      'swarm node doctor' + #13#10 + #13#10 +
      'Already-open terminals must be reopened to see the updated PATH.';
  end;
end;

procedure InitializeWizard;
begin
  PurgeStateCheck := TNewCheckBox.Create(WizardForm);
  PurgeStateCheck.Parent := WizardForm.SelectDirPage;
  PurgeStateCheck.Left := WizardForm.DirEdit.Left;
  PurgeStateCheck.Top := WizardForm.DirEdit.Top + WizardForm.DirEdit.Height + ScaleY(16);
  PurgeStateCheck.Width := WizardForm.DirEdit.Width;
  PurgeStateCheck.Height := ScaleY(34);
  PurgeStateCheck.Caption :=
    'Purge cluster identity, trust, model cache, and evidence when uninstalling';
  PurgeStateCheck.Checked := False;
end;

function UninstallPurgeFlag: String;
var
  Stored: String;
begin
  if ExpandConstant('{param:PURGESTATE|}') <> '' then
  begin
    if Lowercase(ExpandConstant('{param:PURGESTATE|}')) = '1' then
      Result := ' --purge-state'
    else
      Result := '';
  end
  else if RegQueryStringValue(HKCU, 'Software\SwarmInference',
    'PurgeStateOnUninstall', Stored) and (Stored = '1') then
    Result := ' --purge-state'
  else
    Result := '';
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ExitCode: Integer;
  Parameters: String;
  LogPath: String;
begin
  if CurUninstallStep = usUninstall then
  begin
    LogPath := ExpandConstant('{app}\logs\bootstrapper-uninstall.log');
    Parameters := 'uninstall --install-root ' + QuoteArgument(ExpandConstant('{app}')) +
      ' --timeout-seconds 300 --json --log ' + QuoteArgument(LogPath) + UninstallPurgeFlag;
    if not Exec(ExpandConstant('{app}\bin\SwarmBootstrap.exe'), Parameters, '', SW_HIDE,
      ewWaitUntilTerminated, ExitCode) then
      RaiseException('Could not start the native uninstaller engine. Log: ' + LogPath);
    if ExitCode <> 0 then
      RaiseException('Native uninstall failed with exit code ' + IntToStr(ExitCode) +
        '. Application files were not reported as safely removed. Log: ' + LogPath);
  end;
end;
