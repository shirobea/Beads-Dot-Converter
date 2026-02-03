# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('ColorPallet.csv', '.'), ('ui/settings.json', 'ui'), ('ui/window_state.json', 'ui')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
# OpenGL の古い VC9 依存 DLL（未使用）を除外して警告を抑制
_exclude_opengl_vc9 = {
    "freeglut32.vc9.dll",
    "freeglut64.vc9.dll",
    "gle32.vc9.dll",
    "gle64.vc9.dll",
}
a.binaries = [
    b
    for b in a.binaries
    if not any(name in str(b[0]).lower() or name in str(b[1]).lower() for name in _exclude_opengl_vc9)
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ColorChanger',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
