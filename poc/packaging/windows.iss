; Instalador Windows (Inno Setup 6). Compilar com:
;   iscc /DAppVersion=0.1.0 poc\packaging\windows.iss
; Espera o build do PyInstaller em poc\dist\NAM Chain Capture\.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppName "NAM Chain Capture"
#define AppExe "NAM Chain Capture.exe"

[Setup]
AppId={{6F2B8C1E-4E0A-4B7B-9C55-2B1C0C5A9D41}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppName}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Instala só para o usuário atual: não pede senha de administrador.
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=NAM-Chain-Capture-{#AppVersion}-Windows-Setup
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\{#AppName}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

; Capturas, treinos e o pacote de GPU ficam em %APPDATA%\NAM Chain Capture
; e não são apagados na desinstalação (o usuário não perde os modelos).
