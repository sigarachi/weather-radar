@echo off
:: Устанавливаем кодовую страницу на UTF-8
chcp 65001 > nul

set TCL_LIBRARY="C:\Python312\tcl\tcl8.6"
set TK_LIBRARY="C:\Python312\tcl\tk8.6"
fastapi dev main.py