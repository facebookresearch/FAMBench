# The driver can run either comms.par or comms.py.
# To build comms.par:
cd /home/${SUDO_USER}/fbsource/fbcode/param_bench/train/comms/pt
buck build @mode/opt //param_bench/train/comms/pt:comms --show-full-output --out ~/
cp ~/comms.par <proxyworkloads location>/param/train/comms/pt/comms.par

# Note
If proxyworkloads/param is empty, move to that directory and run:
git submodule update --init --recursive
