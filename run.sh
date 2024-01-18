#!/bin/bash

generate_compress_graph(){
    
    echo '生成cfg'   
    python ./graph/cfg.py --isSave
    echo '生成cg'   
    python ./graph/cg.py --isSave
    echo '生成compress graph'   
    python ./graph/combineGraph.py --isSave
}

# vuln="access_control"
generate_compress_graph 
