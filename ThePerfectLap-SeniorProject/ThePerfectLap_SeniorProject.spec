# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ThePerfectLap_SeniorProject.py'],
    pathex=[],
    binaries=[],
    datas=[('assets', 'assets'), ('F1.wav', '.')],
    hiddenimports=['fastf1', 'pandas', 'matplotlib'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'tensorflow', 'tensorboard'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ThePerfectLap_SeniorProject',
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
