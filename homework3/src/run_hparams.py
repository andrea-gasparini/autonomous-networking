import os
import argparse
from typing import Generator, Dict
from rich.console import Console
import time


c = Console()


class RunHParmasTuning:
    def __init__(self, **kargs) -> None:
        self.e_values     = kargs["e_values"]
        self.a_values     = kargs["a_values"]
        self.tt_values    = kargs["tt_values"]
        self.routing_algo = kargs["routing_algo"]
        self.dest_algo    = kargs["dest_algo"]

    def _do_combinations(self) -> Generator:
        for e in self.e_values:
            for a in self.a_values:
                for tt in self.tt_values:
                    yield (e, a, tt)

    def run(self) -> None:
        lines_to_change: Dict[int, str] = {
            12: "    EPSILON = %s\n",
            14: "    AVG_PCKT_THRESHOLD = %s\n",
            16: "    TOTAL_TIME_AVG_PCKT_THRESHOLD = %s\n"
        }

        for epsilon, avg_pkct, tt_avg_pckt in self._do_combinations():
            start = time.time()
            c.print(f"\nNEW RUN: epsilon={epsilon}, avg_pckt={avg_pkct}, total_time_avg_pckt={tt_avg_pckt}\n")

            with open(self.routing_algo, mode="r") as fstream:
                lines = fstream.readlines()
                lines[12] = lines_to_change[12] % epsilon
                lines[14] = lines_to_change[14] % avg_pkct
                lines[16] = lines_to_change[16] % tt_avg_pckt
                lines[10] = "class AiTwoRoutingPrime(BASE_routing):\n"

                new_file_content = ''.join(lines)

            with open(self.dest_algo, mode="w") as wstream:
                wstream.write(new_file_content)
            
            os.system("python src/experiments/run_exp.py")
            end = time.time()
            c.print(f"\nFINISHED RUN: epsilon={epsilon}, avg_pckt={avg_pkct}, total_time_avg_pckt={tt_avg_pckt} in TIME: {end - start}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epsilon-values",           help="Change the values for epsilon",           nargs="+", required=True)
    parser.add_argument("-a", "--avg-pckt-values",          help="Change the values for avg_packets",       nargs="+", required=True)
    parser.add_argument("-t", "--tot-time-avg-pckt-values", help="Change the values for tot_time_avg_pckt", nargs="+", required=True)
    parser.add_argument("-r", "--routing-algo",             help="rel path to the routing algo file",       type=str,  required=True)
    parser.add_argument("-d", "--dest-algo",                help="rel path to the new routing algo file",   type=str,  required=True)

    args = parser.parse_args()

    epsilon_values     = args.epsilon_values
    avg_pckt_values    = args.avg_pckt_values
    tt_avg_pckt_values = args.tot_time_avg_pckt_values
    routing_algo       = os.path.abspath(args.routing_algo)
    dest_algo          = os.path.abspath(args.dest_algo)

    assert os.path.isfile(routing_algo) and os.path.isfile(dest_algo)

    hparams = RunHParmasTuning(e_values=epsilon_values, 
                               a_values=avg_pckt_values,
                               tt_values=tt_avg_pckt_values,
                               routing_algo=routing_algo,
                               dest_algo=dest_algo
                            )

    hparams.run()


if __name__ == "__main__":
    main()