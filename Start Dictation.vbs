' Double-click to start the dictation bar (no console window).
Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
pyw = dir & "\.venv\Scripts\pythonw.exe"
If Not fso.FileExists(pyw) Or Not fso.FileExists(dir & "\cuda_storage\llm\quill-4b-Q4_K_M.gguf") Then
    MsgBox "Not set up yet. Run setup.bat first.", vbExclamation, "Dictation"
    WScript.Quit 1
End If
CreateObject("WScript.Shell").Run """" & pyw & """ """ & dir & "\dictation.py""", 0, False
