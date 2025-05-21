# A Conflict-Driven Approach for Reaching Goals Specified with Negation as Failure

## Running the code
To obtain results shown in Table 1, use the run.sh script.

`sh run.sh <domain> <goal number> <goal type> <specialization operator> <redo>`\

`<domain>`: Rubik's cube (`cube3`) or the 24-puzzle (`puzzle24`)\
`<goal number>`: Corresponds to the goal numbers shown in Table 1\
`<goal type>`: Monotonic specification (`mono`) or a non-monotonic specification that makes use of negation as failure (`nonmono`)\
`<specialization operator>`: Random (`rand`) or conflict-driven (`conf`). Note, this distinction only matters when the goal type is non-monotonic.\
`<redo>`: If `1`, previously completed problem instances are re-done.

### Rubik's cube
#### Goal 1
##### All stickers on the white face are different than the center sticker.
`sh run.sh cube3 1 mono rand 0`
##### There does not exist a sticker on the white face that matches the center sticker.
`sh run.sh cube3 1 nonmono rand 0`\
`sh run.sh cube3 1 nonmono conf 0`

#### Goal 2
##### For all faces, all stickers on that face are different than its center sticker.
`sh run.sh cube3 2 mono rand 0`

##### There does not exist a face that has a sticker that matches its center sticker.
`sh run.sh cube3 2 nonmono rand 0`\
`sh run.sh cube3 2 nonmono conf 0`


### 24-puzzle

#### Goal 1
##### The sum of row 0 is even.
`sh run.sh puzzle24 1 mono rand 0`

##### It is not true the sum of row 0 is odd.
`sh run.sh puzzle24 1 nonmono rand 0`\
`sh run.sh puzzle24 1 nonmono conf 0`

#### Goal 2
##### The sum of all rows is even.
`sh run.sh puzzle24 2 mono rand 0`

##### There does not exist a row whose sum is odd.
`sh run.sh puzzle24 2 nonmono rand 0`\
`sh run.sh puzzle24 2 nonmono conf 0`
