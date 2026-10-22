# Run in an ADMINISTRATOR PowerShell, then reboot. Moves the Windows pagefile from C: to D: (fixed 16 GB).
$cs = Get-CimInstance Win32_ComputerSystem
if ($cs.AutomaticManagedPagefile) { Set-CimInstance -InputObject $cs -Property @{AutomaticManagedPagefile = $false} }
New-CimInstance -ClassName Win32_PageFileSetting -Property @{Name = 'D:\pagefile.sys'} -ErrorAction SilentlyContinue | Out-Null
Get-CimInstance Win32_PageFileSetting -Filter "Name='D:\\pagefile.sys'" |
    Set-CimInstance -Property @{InitialSize = 16384; MaximumSize = 16384}
Get-CimInstance Win32_PageFileSetting -Filter "Name='C:\\pagefile.sys'" | Remove-CimInstance -ErrorAction SilentlyContinue
Get-CimInstance Win32_PageFileSetting | Select-Object Name, InitialSize, MaximumSize
Write-Host "Done. Reboot now for it to take effect."
