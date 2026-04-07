Option Explicit

Dim oShell, oFS
Dim projectDir, pythonw, mainPy, cmd

Set oShell = CreateObject("WScript.Shell")
Set oFS    = CreateObject("Scripting.FileSystemObject")

projectDir = oFS.GetParentFolderName(WScript.ScriptFullName)

pythonw = "C:\Users\sansy\AppData\Local\Programs\Python\Python312\pythonw.exe"
If Not oFS.FileExists(pythonw) Then
    pythonw = "pythonw.exe"
End If

mainPy = projectDir & "\main.py"
If Not oFS.FileExists(mainPy) Then
    MsgBox "main.py not found:" & vbCrLf & mainPy, vbCritical, "Beads Dot Converter"
    WScript.Quit 1
End If

oShell.CurrentDirectory = projectDir
cmd = """" & pythonw & """ """ & mainPy & """"
oShell.Run cmd, 0, False