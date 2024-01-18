#!/bin/bash

generate_compress_graph(){
    
    echo '生成cfg'   
    python ./graph/cfg.py --isSave --vuln_type=${1}
    echo '生成cg'   
    python ./graph/cg.py --isSave --vuln_type=${1}
    echo '生成compress graph'   
    python ./graph/combineGraph.py --isSave --vuln_type=${1}
}

vuln="front_running"
generate_compress_graph ${vuln}
