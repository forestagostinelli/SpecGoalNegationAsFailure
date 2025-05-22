env=$1
goalnum=$2
goaltype=$3
specop=$4
redo=$5
results_dir=$6
heur=$7

time_limit=500
ub_pat=5
expand_size=10

if [[ "$env" == "cube3" ]]; then
  bk_add="specifications/cube3.lp"
  if [[ $goalnum == 1 ]]; then
    if [[ "$goaltype" == "mono" ]]; then
      goal="all_wf_diff_ctr"
    elif [[ "$goaltype" == "nonmono" ]]; then
	    goal="not_exist_wf_mctr"
    fi
  fi

  if [[ $goalnum == 2 ]]; then
    if [[ "$goaltype" == "mono" ]]; then
      goal="all_face_all_diff_ctr"
    elif [[ "$goaltype" == "nonmono" ]]; then
	    goal="not_exist_face_exist_match_ctr"
    fi
  fi
elif [[ "$env" == "puzzle24" ]]; then
  bk_add="specifications/puzzle_sliding.lp"
  if [[ $goalnum == 1 ]]; then
    if [[ "$goaltype" == "mono" ]]; then
      goal="r0_sum_even"
    elif [[ "$goaltype" == "nonmono" ]]; then
	    goal="not_r0_sum_odd"
    fi
  fi

  if [[ $goalnum == 2 ]]; then
    if [[ "$goaltype" == "mono" ]]; then
      goal="all_r_sum_even"
    elif [[ "$goaltype" == "nonmono" ]]; then
	    goal="not_any_r_sum_odd"
    fi
  fi
fi


COMMAND="python run_spec_goal.py --env ${env} --spec \"goal :- ${goal}\" --bk_add $bk_add --states data/${env}/test/spec_asp.pkl --heur $heur --expand_size $expand_size --time_limit $time_limit --ub_pat $ub_pat"


if [[ "$specop" == "rand" ]]; then
  COMMAND="$COMMAND --refine_rand --results ${results_dir}/${env}/${goal}_rand/"
elif [[ "$specop" == "conf" ]]; then
  COMMAND="$COMMAND --refine_conf --refine_confkeep --results ${results_dir}/${env}/${goal}_conf/"
fi

if [[ "$redo" -eq 1 ]]; then
  COMMAND="$COMMAND --redo"
fi

echo $COMMAND
eval $COMMAND
