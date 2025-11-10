# ==========================================================
# Makefile - Topologia 1 core (r0) + 4 bordas (r1..r4)
# LAN = eth1 ; P2P(core) = eth0  em r1..r4
# ==========================================================

ROUTERS = r0 r1 r2 r3 r4

all: dirs configs

dirs:
	@mkdir -p $(ROUTERS)
	@echo "OK: diretórios r0..r4 criados."

# --------- Arquivos padrão (daemons / vtysh.conf) ----------
%/daemons:
	@echo "zebra=yes" > $@
	@echo "bgpd=no" >> $@
	@echo "ospfd=no" >> $@
	@echo "staticd=yes" >> $@
	@echo "✅ daemons -> $(@D)"

%/vtysh.conf:
	@echo "service integrated-vtysh-config" > $@
	@echo "hostname $(@D)" >> $@
	@echo "✅ vtysh.conf -> $(@D)"

configs: \
 r0/daemons r0/vtysh.conf r0/frr.conf \
 r1/daemons r1/vtysh.conf r1/frr.conf \
 r2/daemons r2/vtysh.conf r2/frr.conf \
 r3/daemons r3/vtysh.conf r3/frr.conf \
 r4/daemons r4/vtysh.conf r4/frr.conf

# ----------------- r0 (core) -----------------
r0/frr.conf:
	@echo "hostname r0" > $@
	@echo "service integrated-vtysh-config" >> $@
	@echo "!" >> $@
	@echo "interface r0-eth1" >> $@
	@echo " ip address 10.255.1.1/31" >> $@
	@echo "interface r0-eth2" >> $@
	@echo " ip address 10.255.2.1/31" >> $@
	@echo "interface r0-eth3" >> $@
	@echo " ip address 10.255.3.1/31" >> $@
	@echo "interface r0-eth4" >> $@
	@echo " ip address 10.255.4.1/31" >> $@
	@echo "!" >> $@
	@echo "ip route 10.1.0.0/24 10.255.1.0" >> $@
	@echo "ip route 10.2.0.0/24 10.255.2.0" >> $@
	@echo "ip route 10.3.0.0/24 10.255.3.0" >> $@
	@echo "ip route 10.4.0.0/24 10.255.4.0" >> $@
	@echo "✅ frr.conf -> r0"

# ----------------- r1 (LAN=eth1 ; P2P=eth0) -----------------
r1/frr.conf:
	@echo "hostname r1" > $@
	@echo "service integrated-vtysh-config" >> $@
	@echo "interface r1-eth1" >> $@
	@echo " ip address 10.1.0.1/24" >> $@
	@echo "interface r1-eth0" >> $@
	@echo " ip address 10.255.1.0/31" >> $@
	@echo "!" >> $@
	@echo "ip route 0.0.0.0/0 10.255.1.1" >> $@
	@echo "✅ frr.conf -> r1"

# ----------------- r2 (LAN=eth1 ; P2P=eth0) -----------------
r2/frr.conf:
	@echo "hostname r2" > $@
	@echo "service integrated-vtysh-config" >> $@
	@echo "interface r2-eth1" >> $@
	@echo " ip address 10.2.0.1/24" >> $@
	@echo "interface r2-eth0" >> $@
	@echo " ip address 10.255.2.0/31" >> $@
	@echo "!" >> $@
	@echo "ip route 0.0.0.0/0 10.255.2.1" >> $@
	@echo "✅ frr.conf -> r2"

# ----------------- r3 (LAN=eth1 ; P2P=eth0) -----------------
r3/frr.conf:
	@echo "hostname r3" > $@
	@echo "service integrated-vtysh-config" >> $@
	@echo "interface r3-eth1" >> $@
	@echo " ip address 10.3.0.1/24" >> $@
	@echo "interface r3-eth0" >> $@
	@echo " ip address 10.255.3.0/31" >> $@
	@echo "!" >> $@
	@echo "ip route 0.0.0.0/0 10.255.3.1" >> $@
	@echo "✅ frr.conf -> r3"

# ----------------- r4 (LAN=eth1 ; P2P=eth0) -----------------
r4/frr.conf:
	@echo "hostname r4" > $@
	@echo "service integrated-vtysh-config" >> $@
	@echo "interface r4-eth1" >> $@
	@echo " ip address 10.4.0.1/24" >> $@
	@echo "interface r4-eth0" >> $@
	@echo " ip address 10.255.4.0/31" >> $@
	@echo "!" >> $@
	@echo "ip route 0.0.0.0/0 10.255.4.1" >> $@
	@echo "✅ frr.conf -> r4"

# ----------------- utilitários -----------------
clean:
	@rm -rf $(ROUTERS)
	@echo "🧹 limpo."

run:
	@sudo python3 topo.py

stop:
	@sudo mn -c
