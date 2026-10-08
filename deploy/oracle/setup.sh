#!/usr/bin/env bash
# Run on the Oracle Ubuntu VM, inside the unpacked bundle folder (~/smartroute).
# Usage: bash setup.sh            -> serves on http://<public-ip>.sslip.io with automatic HTTPS
set -euo pipefail
cd "$(dirname "$0")"

# 1. Docker
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
fi

# 2. Open ports 80/443 in the VM firewall (Oracle's Ubuntu image blocks them by default)
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save 2>/dev/null || true

# 3. Secrets file (edit .env afterwards to add API keys, then: sudo docker compose up -d)
[ -f .env ] || cat > .env <<EOF
SECRET_KEY=$(openssl rand -hex 32)
WEB_CONCURRENCY=2
# SERPAPI_KEY=
# WEATHERAPI_KEY=
# GEOAPIFY_API_KEY=
# FOURSQUARE_API_KEY=
# AVIATIONSTACK_API_KEY=
# OPENSKY_CLIENT_ID=
# OPENSKY_CLIENT_SECRET=
# ADMIN_EMAILS=you@example.com
EOF

# 4. Public hostname: <ip>.sslip.io resolves to the IP, so Caddy can get a free HTTPS cert
IP=$(curl -s https://ifconfig.me)
echo "DOMAIN=${IP//./-}.sslip.io" > .domain
sudo docker compose --env-file .env --env-file .domain up -d --build
echo "Done. Open: https://$(cut -d= -f2 .domain)"
