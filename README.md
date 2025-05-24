# A Conflict-Driven Approach for Reaching Goals Specified with Negation as Failure

## Installation

`pip install deepxube==0.1.6`\
`conda install pytorch==2.3.1 -c pytorch`  (Needs to be compatible with numpy==2.0.0. pytorch==2.5.1 could also work)\
`pip install numpy==2.0.0`

## Trained heuristic functions
You can download the trained heuristic functions [here](https://drive.proton.me/urls/ZGQ3XKD2DC#w6s4sIuxF70c).\
Use the `current.pt` file.

## Running the code
To obtain results shown in Table 1, use the run.sh script. The `results_paper` folder shows the outputs obtained for Table 1. 

`sh run.sh <domain> <goal number> <goal type> <specialization operator> <redo> <dir> <heur>`

`<domain>`: Rubik's cube (`cube3`) or the 24-puzzle (`puzzle24`)\
`<goal number>`: Corresponds to the goal numbers shown in Table 1\
`<goal type>`: Monotonic specification (`mono`) or a non-monotonic specification that makes use of negation as failure (`nonmono`)\
`<specialization operator>`: Random (`rand`) or conflict-driven (`conf`). Note, this distinction only matters when the goal type is non-monotonic.\
`<redo>`: If `1`, previously completed problem instances are re-done.\
`<dir>`: Output directory for results\
`<heur>`: Location of heuristic function. For example, `models/cube3/current.pt`.

### Rubik's cube
#### Goal 1
##### All stickers on the white face are different than the center sticker.
`sh run.sh cube3 1 mono rand 0 results <heur>`
##### There does not exist a sticker on the white face that matches the center sticker.
`sh run.sh cube3 1 nonmono rand 0 results <heur>`\
`sh run.sh cube3 1 nonmono conf 0 results <heur>`

#### Goal 2
##### For all faces, all stickers on that face are different than its center sticker.
`sh run.sh cube3 2 mono rand 0 results <heur>`

##### There does not exist a face that has a sticker that matches its center sticker.
`sh run.sh cube3 2 nonmono rand 0 results <heur>`\
`sh run.sh cube3 2 nonmono conf 0 results <heur>`


### 24-puzzle

#### Goal 1
##### The sum of row 0 is even.
`sh run.sh puzzle24 1 mono rand 0 results <heur>`

##### It is not true the sum of row 0 is odd.
`sh run.sh puzzle24 1 nonmono rand 0 results <heur>`\
`sh run.sh puzzle24 1 nonmono conf 0 results <heur>`

#### Goal 2
##### The sum of all rows is even.
`sh run.sh puzzle24 2 mono rand 0 results <heur>`

##### There does not exist a row whose sum is odd.
`sh run.sh puzzle24 2 nonmono rand 0 results <heur>`\
`sh run.sh puzzle24 2 nonmono conf 0 results <heur>`
