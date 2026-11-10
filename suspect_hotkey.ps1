# Глобальная горячая клавиша Ctrl+Alt+F9: «услышал странную реплику в игре» → студия отмечает реплики за последние 5 минут на проверку (suspect.log).
# Запуск в сессии пользователя (видимый ввод): powershell -ExecutionPolicy Bypass -File suspect_hotkey.ps1  (или задачей планировщика /it при входе)
Add-Type -TypeDefinition @"
using System; using System.Runtime.InteropServices;
public class HK { [DllImport("user32.dll")] public static extern bool RegisterHotKey(IntPtr h, int id, uint mod, uint vk);
 [DllImport("user32.dll")] public static extern int GetMessage(out MSG m, IntPtr h, uint a, uint b);
 [StructLayout(LayoutKind.Sequential)] public struct MSG { public IntPtr hwnd; public uint message; public IntPtr wParam, lParam; public uint time; public int x, y; } }
"@
$minutes = 5; $url = "http://127.0.0.1:8777/api/suspect"
# MOD_ALT=1, MOD_CONTROL=2; VK_F9=0x78
if (-not [HK]::RegisterHotKey([IntPtr]::Zero, 1, 3, 0x78)) { Write-Host "Ctrl+Alt+F9 уже занята"; exit 1 }
Write-Host "Ждём Ctrl+Alt+F9 (странная реплика → последние $minutes мин на проверку)"
$m = New-Object HK+MSG
while ([HK]::GetMessage([ref]$m, [IntPtr]::Zero, 0, 0) -gt 0) {
  if ($m.message -eq 0x312) {
    try { $r = Invoke-RestMethod -Method Post -Uri $url -ContentType "application/json" -Body (@{minutes=$minutes} | ConvertTo-Json); [console]::Beep(880,150); Write-Host "$(Get-Date -f HH:mm:ss) отмечено: $($r.n) ($($r.note))" }
    catch { [console]::Beep(220,400); Write-Host "ошибка: $_" }
  }
}
