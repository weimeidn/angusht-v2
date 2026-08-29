@echo off
chcp 65001 >nul
title Angusht v2.4 — Нейроморфная Когнитивная Система

echo.
echo  ========================================================
echo     Angusht v2.4 — Нейроморфная Когнитивная Система
echo     6 ядер x 216K LIF-нейронов = 1 296 000
echo     Веб-поиск + Самообучение + STDP
echo  ========================================================
echo.

where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [!] Node.js не найден. Установите: https://nodejs.org
    pause
    exit /b 1
)

echo [*] Запуск Angusht v2.4 на http://localhost:3000
echo [*] Нажмите Ctrl+C для остановки
echo.

if not exist "node_modules" (
    echo [*] Установка зависимостей...
    call npm install
    echo.
)

start http://localhost:3000
call npm run dev
