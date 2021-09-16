processes=2
processes_per_node=2
hostfile=
beginsize=8
endsize='256M'
steps=100
nonblocking=1
collective='all-to-all'
backend='mpi' #can be "gloo", "nccl" or "xla"
device='cpu' # can be "cuda"


usage() { echo "Usage: $0 [-p <#processes>] [-n <processors_per_node>] [-h hostfile] [-s <steps>] [-o <0 = blocking, 1 = non-blocking>] [-b <'mpi'|'gloo'|'nccl'|'xla'>]  [-d <'cpu'|'cuda'>]"; exit 1; }

while getopts "p:n:h:s:o:b:d:" flag
do
    case "${flag}" in
        p) processes=${OPTARG};;
        n) processes_per_node=${OPTARG};;
        h) hostfile=${OPTARG};;
        s) steps=${OPTARG};;
        o) nonblocking=${OPTARG};;
        b) backend=${OPTARG};;
        d) device=${OPTARG};;
    esac
done
shift $((OPTIND-1))

if [ -z "$hostfile" ]
then
  mpirun -np $processes -N $processes_per_node python ../../../../param/train/comms/pt/comms.py --master_ip 127.0.0.1 --b 8 --e 256M --n $steps --f 2 --z $nonblocking --collective $collective --backedn $backend --device $device --log INFO
else
  mpirun -np $processes -N $processes_per_node --hostfile $hostfile python ../../../../param/train/comms/pt/comms.py --master_ip $(head -n 1 $hostfile) --b 8 --e 256M --n $steps --f 2 --z $nonblocking --collective $collective --backedn $backend --device $device --log INFO
fi
