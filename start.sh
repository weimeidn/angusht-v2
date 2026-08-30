#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Angusht v2.4 — Нейроморфная когнитивная архитектура
# Запуск одним кликом: chmod +x start.sh && ./start.sh
# ═══════════════════════════════════════════════════════════

echo ""
echo "  ╔════════════════════════════════════════════════════════╗"
echo "  ║     Angusht v2.4 — Нейроморфная Когнитивная Система    ║"
echo "  ║     6 ядер x 216K LIF-нейронов = 1 296 000             ║"
echo "  ║     Веб-поиск + Самообучение + STDP                    ║"
echo "  ╚════════════════════════════════════════════════════════╝"
echo ""

# Проверка Node.js
if ! command -v node &> /dev/null; then
    echo "[!] Node.js не найден. Установите: https://nodejs.org"
    exit 1
fi

echo "[*] Node.js: $(node --version)"

# Установка зависимостей (если нет node_modules)
if [ ! -d "node_modules" ]; then
    echo "[*] Установка зависимостей..."
    npm install
fi

# Запуск
echo "[*] Запуск Angusht v2.4 на http://localhost:3000"
echo "[*] Нажмите Ctrl+C для остановки"
echo ""

# Открываем браузер через 3 секунды (только одну вкладку, в зависимости от ОС)
(
    sleep 3
    if command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:3000
    elif command -v open &> /dev/null; then
        open http://localhost:3000
    fi
) &

npm run dev
