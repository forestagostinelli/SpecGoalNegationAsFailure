% Colors.
color(w).
color(y).
color(o).
color(r).
color(b).
color(g).
dif_col(Col1, Col2) :- color(Col1), color(Col2), Col1 != Col2.

white(w).
yellow(y).
orange(o).
red(r).
blue(b).
green(g).
% Cubelets.
center_cbl(w_c).
center_cbl(y_c).
center_cbl(o_c).
center_cbl(r_c).
center_cbl(b_c).
center_cbl(g_c).
edge_cbl(wo_c).
edge_cbl(wg_c).
edge_cbl(wb_c).
edge_cbl(wr_c).
edge_cbl(yo_c).
edge_cbl(yb_c).
edge_cbl(yg_c).
edge_cbl(yr_c).
edge_cbl(ob_c).
edge_cbl(og_c).
edge_cbl(rg_c).
edge_cbl(rb_c).
corner_cbl(wog_c).
corner_cbl(wob_c).
corner_cbl(wrg_c).
corner_cbl(wrb_c).
corner_cbl(yob_c).
corner_cbl(yog_c).
corner_cbl(yrb_c).
corner_cbl(yrg_c).
cubelet(Cbl) :- center_cbl(Cbl).
cubelet(Cbl) :- edge_cbl(Cbl).
cubelet(Cbl) :- corner_cbl(Cbl).
dif_cbl(Cbl1, Cbl2) :- cubelet(Cbl1), cubelet(Cbl2), Cbl1 != Cbl2.
% has_stk_col.
has_stk_col(w_c, w).
has_stk_col(y_c, y).
has_stk_col(o_c, o).
has_stk_col(r_c, r).
has_stk_col(b_c, b).
has_stk_col(g_c, g).
has_stk_col(wo_c, w).
has_stk_col(wo_c, o).
has_stk_col(wg_c, w).
has_stk_col(wg_c, g).
has_stk_col(wb_c, w).
has_stk_col(wb_c, b).
has_stk_col(wr_c, w).
has_stk_col(wr_c, r).
has_stk_col(yo_c, y).
has_stk_col(yo_c, o).
has_stk_col(yb_c, y).
has_stk_col(yb_c, b).
has_stk_col(yg_c, y).
has_stk_col(yg_c, g).
has_stk_col(yr_c, y).
has_stk_col(yr_c, r).
has_stk_col(ob_c, o).
has_stk_col(ob_c, b).
has_stk_col(og_c, o).
has_stk_col(og_c, g).
has_stk_col(rg_c, r).
has_stk_col(rg_c, g).
has_stk_col(rb_c, r).
has_stk_col(rb_c, b).
has_stk_col(wog_c, w).
has_stk_col(wog_c, o).
has_stk_col(wog_c, g).
has_stk_col(wob_c, w).
has_stk_col(wob_c, o).
has_stk_col(wob_c, b).
has_stk_col(wrg_c, w).
has_stk_col(wrg_c, r).
has_stk_col(wrg_c, g).
has_stk_col(wrb_c, w).
has_stk_col(wrb_c, r).
has_stk_col(wrb_c, b).
has_stk_col(yob_c, y).
has_stk_col(yob_c, o).
has_stk_col(yob_c, b).
has_stk_col(yog_c, y).
has_stk_col(yog_c, o).
has_stk_col(yog_c, g).
has_stk_col(yrb_c, y).
has_stk_col(yrb_c, r).
has_stk_col(yrb_c, b).
has_stk_col(yrg_c, y).
has_stk_col(yrg_c, r).
has_stk_col(yrg_c, g).
% Directions.
direction(cl).
direction(cc).
direction(op).
dif_dir(Dir1, Dir2) :- direction(Dir1), direction(Dir2), Dir1 != Dir2.

clockwise(cl).
counterclockwise(cc).
opposite(op).
% Faces.
face(w_f).
face(y_f).
face(o_f).
face(r_f).
face(b_f).
face(g_f).
dif_face(F1, F2) :- face(F1), face(F2), F1 != F2.

face_col(w_f, w).
face_col(y_f, y).
face_col(o_f, o).
face_col(r_f, r).
face_col(b_f, b).
face_col(g_f, g).

face_rel(w_f, b_f, r_f, cl).
face_rel(w_f, r_f, g_f, cl).
face_rel(w_f, g_f, o_f, cl).
face_rel(w_f, o_f, b_f, cl).
face_rel(o_f, y_f, b_f, cl).
face_rel(o_f, b_f, w_f, cl).
face_rel(o_f, w_f, g_f, cl).
face_rel(o_f, g_f, y_f, cl).
face_rel(g_f, o_f, w_f, cl).
face_rel(g_f, w_f, r_f, cl).
face_rel(g_f, r_f, y_f, cl).
face_rel(g_f, y_f, o_f, cl).
face_rel(y_f, b_f, o_f, cl).
face_rel(y_f, r_f, b_f, cl).
face_rel(y_f, g_f, r_f, cl).
face_rel(y_f, o_f, g_f, cl).
face_rel(r_f, y_f, g_f, cl).
face_rel(r_f, b_f, y_f, cl).
face_rel(r_f, w_f, b_f, cl).
face_rel(r_f, g_f, w_f, cl).
face_rel(b_f, o_f, y_f, cl).
face_rel(b_f, w_f, o_f, cl).
face_rel(b_f, r_f, w_f, cl).
face_rel(b_f, y_f, r_f, cl).

face_rel(A, B, C, cc) :- face_rel(A, C, B, cl).
face_rel(A, B, D, op) :- face_rel(A, B, C, cl), face_rel(A, C, D, cl).
face_rel(A, D, B, op) :- face_rel(A, B, C, cl), face_rel(A, C, D, cl).
face_adj(F0, F1) :- face_rel(F0, _, F1, _).
% State dependent predicates.
index(0..53).
-at_idx(Col, I) :- at_idx(Col2, I), color(Col), color(Col2), not Col=Col2.
% onface.
onface(w_c, w, w_f) :- at_idx(w, 4).
onface(y_c, y, w_f) :- at_idx(y, 4).
onface(o_c, o, w_f) :- at_idx(o, 4).
onface(r_c, r, w_f) :- at_idx(r, 4).
onface(b_c, b, w_f) :- at_idx(b, 4).
onface(g_c, g, w_f) :- at_idx(g, 4).
onface(w_c, w, y_f) :- at_idx(w, 13).
onface(y_c, y, y_f) :- at_idx(y, 13).
onface(o_c, o, y_f) :- at_idx(o, 13).
onface(r_c, r, y_f) :- at_idx(r, 13).
onface(b_c, b, y_f) :- at_idx(b, 13).
onface(g_c, g, y_f) :- at_idx(g, 13).
onface(w_c, w, o_f) :- at_idx(w, 22).
onface(y_c, y, o_f) :- at_idx(y, 22).
onface(o_c, o, o_f) :- at_idx(o, 22).
onface(r_c, r, o_f) :- at_idx(r, 22).
onface(b_c, b, o_f) :- at_idx(b, 22).
onface(g_c, g, o_f) :- at_idx(g, 22).
onface(w_c, w, r_f) :- at_idx(w, 31).
onface(y_c, y, r_f) :- at_idx(y, 31).
onface(o_c, o, r_f) :- at_idx(o, 31).
onface(r_c, r, r_f) :- at_idx(r, 31).
onface(b_c, b, r_f) :- at_idx(b, 31).
onface(g_c, g, r_f) :- at_idx(g, 31).
onface(w_c, w, b_f) :- at_idx(w, 40).
onface(y_c, y, b_f) :- at_idx(y, 40).
onface(o_c, o, b_f) :- at_idx(o, 40).
onface(r_c, r, b_f) :- at_idx(r, 40).
onface(b_c, b, b_f) :- at_idx(b, 40).
onface(g_c, g, b_f) :- at_idx(g, 40).
onface(w_c, w, g_f) :- at_idx(w, 49).
onface(y_c, y, g_f) :- at_idx(y, 49).
onface(o_c, o, g_f) :- at_idx(o, 49).
onface(r_c, r, g_f) :- at_idx(r, 49).
onface(b_c, b, g_f) :- at_idx(b, 49).
onface(g_c, g, g_f) :- at_idx(g, 49).
onface(wo_c, w, w_f) :- at_idx(w, 1), at_idx(o, 23).
onface(wo_c, o, o_f) :- at_idx(w, 1), at_idx(o, 23).
onface(wo_c, o, w_f) :- at_idx(o, 1), at_idx(w, 23).
onface(wo_c, w, o_f) :- at_idx(o, 1), at_idx(w, 23).
onface(wg_c, w, w_f) :- at_idx(w, 1), at_idx(g, 23).
onface(wg_c, g, o_f) :- at_idx(w, 1), at_idx(g, 23).
onface(wg_c, g, w_f) :- at_idx(g, 1), at_idx(w, 23).
onface(wg_c, w, o_f) :- at_idx(g, 1), at_idx(w, 23).
onface(wb_c, w, w_f) :- at_idx(w, 1), at_idx(b, 23).
onface(wb_c, b, o_f) :- at_idx(w, 1), at_idx(b, 23).
onface(wb_c, b, w_f) :- at_idx(b, 1), at_idx(w, 23).
onface(wb_c, w, o_f) :- at_idx(b, 1), at_idx(w, 23).
onface(wr_c, w, w_f) :- at_idx(w, 1), at_idx(r, 23).
onface(wr_c, r, o_f) :- at_idx(w, 1), at_idx(r, 23).
onface(wr_c, r, w_f) :- at_idx(r, 1), at_idx(w, 23).
onface(wr_c, w, o_f) :- at_idx(r, 1), at_idx(w, 23).
onface(yo_c, y, w_f) :- at_idx(y, 1), at_idx(o, 23).
onface(yo_c, o, o_f) :- at_idx(y, 1), at_idx(o, 23).
onface(yo_c, o, w_f) :- at_idx(o, 1), at_idx(y, 23).
onface(yo_c, y, o_f) :- at_idx(o, 1), at_idx(y, 23).
onface(yb_c, y, w_f) :- at_idx(y, 1), at_idx(b, 23).
onface(yb_c, b, o_f) :- at_idx(y, 1), at_idx(b, 23).
onface(yb_c, b, w_f) :- at_idx(b, 1), at_idx(y, 23).
onface(yb_c, y, o_f) :- at_idx(b, 1), at_idx(y, 23).
onface(yg_c, y, w_f) :- at_idx(y, 1), at_idx(g, 23).
onface(yg_c, g, o_f) :- at_idx(y, 1), at_idx(g, 23).
onface(yg_c, g, w_f) :- at_idx(g, 1), at_idx(y, 23).
onface(yg_c, y, o_f) :- at_idx(g, 1), at_idx(y, 23).
onface(yr_c, y, w_f) :- at_idx(y, 1), at_idx(r, 23).
onface(yr_c, r, o_f) :- at_idx(y, 1), at_idx(r, 23).
onface(yr_c, r, w_f) :- at_idx(r, 1), at_idx(y, 23).
onface(yr_c, y, o_f) :- at_idx(r, 1), at_idx(y, 23).
onface(ob_c, o, w_f) :- at_idx(o, 1), at_idx(b, 23).
onface(ob_c, b, o_f) :- at_idx(o, 1), at_idx(b, 23).
onface(ob_c, b, w_f) :- at_idx(b, 1), at_idx(o, 23).
onface(ob_c, o, o_f) :- at_idx(b, 1), at_idx(o, 23).
onface(og_c, o, w_f) :- at_idx(o, 1), at_idx(g, 23).
onface(og_c, g, o_f) :- at_idx(o, 1), at_idx(g, 23).
onface(og_c, g, w_f) :- at_idx(g, 1), at_idx(o, 23).
onface(og_c, o, o_f) :- at_idx(g, 1), at_idx(o, 23).
onface(rg_c, r, w_f) :- at_idx(r, 1), at_idx(g, 23).
onface(rg_c, g, o_f) :- at_idx(r, 1), at_idx(g, 23).
onface(rg_c, g, w_f) :- at_idx(g, 1), at_idx(r, 23).
onface(rg_c, r, o_f) :- at_idx(g, 1), at_idx(r, 23).
onface(rb_c, r, w_f) :- at_idx(r, 1), at_idx(b, 23).
onface(rb_c, b, o_f) :- at_idx(r, 1), at_idx(b, 23).
onface(rb_c, b, w_f) :- at_idx(b, 1), at_idx(r, 23).
onface(rb_c, r, o_f) :- at_idx(b, 1), at_idx(r, 23).
onface(wo_c, w, w_f) :- at_idx(w, 3), at_idx(o, 50).
onface(wo_c, o, g_f) :- at_idx(w, 3), at_idx(o, 50).
onface(wo_c, o, w_f) :- at_idx(o, 3), at_idx(w, 50).
onface(wo_c, w, g_f) :- at_idx(o, 3), at_idx(w, 50).
onface(wg_c, w, w_f) :- at_idx(w, 3), at_idx(g, 50).
onface(wg_c, g, g_f) :- at_idx(w, 3), at_idx(g, 50).
onface(wg_c, g, w_f) :- at_idx(g, 3), at_idx(w, 50).
onface(wg_c, w, g_f) :- at_idx(g, 3), at_idx(w, 50).
onface(wb_c, w, w_f) :- at_idx(w, 3), at_idx(b, 50).
onface(wb_c, b, g_f) :- at_idx(w, 3), at_idx(b, 50).
onface(wb_c, b, w_f) :- at_idx(b, 3), at_idx(w, 50).
onface(wb_c, w, g_f) :- at_idx(b, 3), at_idx(w, 50).
onface(wr_c, w, w_f) :- at_idx(w, 3), at_idx(r, 50).
onface(wr_c, r, g_f) :- at_idx(w, 3), at_idx(r, 50).
onface(wr_c, r, w_f) :- at_idx(r, 3), at_idx(w, 50).
onface(wr_c, w, g_f) :- at_idx(r, 3), at_idx(w, 50).
onface(yo_c, y, w_f) :- at_idx(y, 3), at_idx(o, 50).
onface(yo_c, o, g_f) :- at_idx(y, 3), at_idx(o, 50).
onface(yo_c, o, w_f) :- at_idx(o, 3), at_idx(y, 50).
onface(yo_c, y, g_f) :- at_idx(o, 3), at_idx(y, 50).
onface(yb_c, y, w_f) :- at_idx(y, 3), at_idx(b, 50).
onface(yb_c, b, g_f) :- at_idx(y, 3), at_idx(b, 50).
onface(yb_c, b, w_f) :- at_idx(b, 3), at_idx(y, 50).
onface(yb_c, y, g_f) :- at_idx(b, 3), at_idx(y, 50).
onface(yg_c, y, w_f) :- at_idx(y, 3), at_idx(g, 50).
onface(yg_c, g, g_f) :- at_idx(y, 3), at_idx(g, 50).
onface(yg_c, g, w_f) :- at_idx(g, 3), at_idx(y, 50).
onface(yg_c, y, g_f) :- at_idx(g, 3), at_idx(y, 50).
onface(yr_c, y, w_f) :- at_idx(y, 3), at_idx(r, 50).
onface(yr_c, r, g_f) :- at_idx(y, 3), at_idx(r, 50).
onface(yr_c, r, w_f) :- at_idx(r, 3), at_idx(y, 50).
onface(yr_c, y, g_f) :- at_idx(r, 3), at_idx(y, 50).
onface(ob_c, o, w_f) :- at_idx(o, 3), at_idx(b, 50).
onface(ob_c, b, g_f) :- at_idx(o, 3), at_idx(b, 50).
onface(ob_c, b, w_f) :- at_idx(b, 3), at_idx(o, 50).
onface(ob_c, o, g_f) :- at_idx(b, 3), at_idx(o, 50).
onface(og_c, o, w_f) :- at_idx(o, 3), at_idx(g, 50).
onface(og_c, g, g_f) :- at_idx(o, 3), at_idx(g, 50).
onface(og_c, g, w_f) :- at_idx(g, 3), at_idx(o, 50).
onface(og_c, o, g_f) :- at_idx(g, 3), at_idx(o, 50).
onface(rg_c, r, w_f) :- at_idx(r, 3), at_idx(g, 50).
onface(rg_c, g, g_f) :- at_idx(r, 3), at_idx(g, 50).
onface(rg_c, g, w_f) :- at_idx(g, 3), at_idx(r, 50).
onface(rg_c, r, g_f) :- at_idx(g, 3), at_idx(r, 50).
onface(rb_c, r, w_f) :- at_idx(r, 3), at_idx(b, 50).
onface(rb_c, b, g_f) :- at_idx(r, 3), at_idx(b, 50).
onface(rb_c, b, w_f) :- at_idx(b, 3), at_idx(r, 50).
onface(rb_c, r, g_f) :- at_idx(b, 3), at_idx(r, 50).
onface(wo_c, w, w_f) :- at_idx(w, 5), at_idx(o, 41).
onface(wo_c, o, b_f) :- at_idx(w, 5), at_idx(o, 41).
onface(wo_c, o, w_f) :- at_idx(o, 5), at_idx(w, 41).
onface(wo_c, w, b_f) :- at_idx(o, 5), at_idx(w, 41).
onface(wg_c, w, w_f) :- at_idx(w, 5), at_idx(g, 41).
onface(wg_c, g, b_f) :- at_idx(w, 5), at_idx(g, 41).
onface(wg_c, g, w_f) :- at_idx(g, 5), at_idx(w, 41).
onface(wg_c, w, b_f) :- at_idx(g, 5), at_idx(w, 41).
onface(wb_c, w, w_f) :- at_idx(w, 5), at_idx(b, 41).
onface(wb_c, b, b_f) :- at_idx(w, 5), at_idx(b, 41).
onface(wb_c, b, w_f) :- at_idx(b, 5), at_idx(w, 41).
onface(wb_c, w, b_f) :- at_idx(b, 5), at_idx(w, 41).
onface(wr_c, w, w_f) :- at_idx(w, 5), at_idx(r, 41).
onface(wr_c, r, b_f) :- at_idx(w, 5), at_idx(r, 41).
onface(wr_c, r, w_f) :- at_idx(r, 5), at_idx(w, 41).
onface(wr_c, w, b_f) :- at_idx(r, 5), at_idx(w, 41).
onface(yo_c, y, w_f) :- at_idx(y, 5), at_idx(o, 41).
onface(yo_c, o, b_f) :- at_idx(y, 5), at_idx(o, 41).
onface(yo_c, o, w_f) :- at_idx(o, 5), at_idx(y, 41).
onface(yo_c, y, b_f) :- at_idx(o, 5), at_idx(y, 41).
onface(yb_c, y, w_f) :- at_idx(y, 5), at_idx(b, 41).
onface(yb_c, b, b_f) :- at_idx(y, 5), at_idx(b, 41).
onface(yb_c, b, w_f) :- at_idx(b, 5), at_idx(y, 41).
onface(yb_c, y, b_f) :- at_idx(b, 5), at_idx(y, 41).
onface(yg_c, y, w_f) :- at_idx(y, 5), at_idx(g, 41).
onface(yg_c, g, b_f) :- at_idx(y, 5), at_idx(g, 41).
onface(yg_c, g, w_f) :- at_idx(g, 5), at_idx(y, 41).
onface(yg_c, y, b_f) :- at_idx(g, 5), at_idx(y, 41).
onface(yr_c, y, w_f) :- at_idx(y, 5), at_idx(r, 41).
onface(yr_c, r, b_f) :- at_idx(y, 5), at_idx(r, 41).
onface(yr_c, r, w_f) :- at_idx(r, 5), at_idx(y, 41).
onface(yr_c, y, b_f) :- at_idx(r, 5), at_idx(y, 41).
onface(ob_c, o, w_f) :- at_idx(o, 5), at_idx(b, 41).
onface(ob_c, b, b_f) :- at_idx(o, 5), at_idx(b, 41).
onface(ob_c, b, w_f) :- at_idx(b, 5), at_idx(o, 41).
onface(ob_c, o, b_f) :- at_idx(b, 5), at_idx(o, 41).
onface(og_c, o, w_f) :- at_idx(o, 5), at_idx(g, 41).
onface(og_c, g, b_f) :- at_idx(o, 5), at_idx(g, 41).
onface(og_c, g, w_f) :- at_idx(g, 5), at_idx(o, 41).
onface(og_c, o, b_f) :- at_idx(g, 5), at_idx(o, 41).
onface(rg_c, r, w_f) :- at_idx(r, 5), at_idx(g, 41).
onface(rg_c, g, b_f) :- at_idx(r, 5), at_idx(g, 41).
onface(rg_c, g, w_f) :- at_idx(g, 5), at_idx(r, 41).
onface(rg_c, r, b_f) :- at_idx(g, 5), at_idx(r, 41).
onface(rb_c, r, w_f) :- at_idx(r, 5), at_idx(b, 41).
onface(rb_c, b, b_f) :- at_idx(r, 5), at_idx(b, 41).
onface(rb_c, b, w_f) :- at_idx(b, 5), at_idx(r, 41).
onface(rb_c, r, b_f) :- at_idx(b, 5), at_idx(r, 41).
onface(wo_c, w, w_f) :- at_idx(w, 7), at_idx(o, 32).
onface(wo_c, o, r_f) :- at_idx(w, 7), at_idx(o, 32).
onface(wo_c, o, w_f) :- at_idx(o, 7), at_idx(w, 32).
onface(wo_c, w, r_f) :- at_idx(o, 7), at_idx(w, 32).
onface(wg_c, w, w_f) :- at_idx(w, 7), at_idx(g, 32).
onface(wg_c, g, r_f) :- at_idx(w, 7), at_idx(g, 32).
onface(wg_c, g, w_f) :- at_idx(g, 7), at_idx(w, 32).
onface(wg_c, w, r_f) :- at_idx(g, 7), at_idx(w, 32).
onface(wb_c, w, w_f) :- at_idx(w, 7), at_idx(b, 32).
onface(wb_c, b, r_f) :- at_idx(w, 7), at_idx(b, 32).
onface(wb_c, b, w_f) :- at_idx(b, 7), at_idx(w, 32).
onface(wb_c, w, r_f) :- at_idx(b, 7), at_idx(w, 32).
onface(wr_c, w, w_f) :- at_idx(w, 7), at_idx(r, 32).
onface(wr_c, r, r_f) :- at_idx(w, 7), at_idx(r, 32).
onface(wr_c, r, w_f) :- at_idx(r, 7), at_idx(w, 32).
onface(wr_c, w, r_f) :- at_idx(r, 7), at_idx(w, 32).
onface(yo_c, y, w_f) :- at_idx(y, 7), at_idx(o, 32).
onface(yo_c, o, r_f) :- at_idx(y, 7), at_idx(o, 32).
onface(yo_c, o, w_f) :- at_idx(o, 7), at_idx(y, 32).
onface(yo_c, y, r_f) :- at_idx(o, 7), at_idx(y, 32).
onface(yb_c, y, w_f) :- at_idx(y, 7), at_idx(b, 32).
onface(yb_c, b, r_f) :- at_idx(y, 7), at_idx(b, 32).
onface(yb_c, b, w_f) :- at_idx(b, 7), at_idx(y, 32).
onface(yb_c, y, r_f) :- at_idx(b, 7), at_idx(y, 32).
onface(yg_c, y, w_f) :- at_idx(y, 7), at_idx(g, 32).
onface(yg_c, g, r_f) :- at_idx(y, 7), at_idx(g, 32).
onface(yg_c, g, w_f) :- at_idx(g, 7), at_idx(y, 32).
onface(yg_c, y, r_f) :- at_idx(g, 7), at_idx(y, 32).
onface(yr_c, y, w_f) :- at_idx(y, 7), at_idx(r, 32).
onface(yr_c, r, r_f) :- at_idx(y, 7), at_idx(r, 32).
onface(yr_c, r, w_f) :- at_idx(r, 7), at_idx(y, 32).
onface(yr_c, y, r_f) :- at_idx(r, 7), at_idx(y, 32).
onface(ob_c, o, w_f) :- at_idx(o, 7), at_idx(b, 32).
onface(ob_c, b, r_f) :- at_idx(o, 7), at_idx(b, 32).
onface(ob_c, b, w_f) :- at_idx(b, 7), at_idx(o, 32).
onface(ob_c, o, r_f) :- at_idx(b, 7), at_idx(o, 32).
onface(og_c, o, w_f) :- at_idx(o, 7), at_idx(g, 32).
onface(og_c, g, r_f) :- at_idx(o, 7), at_idx(g, 32).
onface(og_c, g, w_f) :- at_idx(g, 7), at_idx(o, 32).
onface(og_c, o, r_f) :- at_idx(g, 7), at_idx(o, 32).
onface(rg_c, r, w_f) :- at_idx(r, 7), at_idx(g, 32).
onface(rg_c, g, r_f) :- at_idx(r, 7), at_idx(g, 32).
onface(rg_c, g, w_f) :- at_idx(g, 7), at_idx(r, 32).
onface(rg_c, r, r_f) :- at_idx(g, 7), at_idx(r, 32).
onface(rb_c, r, w_f) :- at_idx(r, 7), at_idx(b, 32).
onface(rb_c, b, r_f) :- at_idx(r, 7), at_idx(b, 32).
onface(rb_c, b, w_f) :- at_idx(b, 7), at_idx(r, 32).
onface(rb_c, r, r_f) :- at_idx(b, 7), at_idx(r, 32).
onface(wo_c, w, y_f) :- at_idx(w, 10), at_idx(o, 21).
onface(wo_c, o, o_f) :- at_idx(w, 10), at_idx(o, 21).
onface(wo_c, o, y_f) :- at_idx(o, 10), at_idx(w, 21).
onface(wo_c, w, o_f) :- at_idx(o, 10), at_idx(w, 21).
onface(wg_c, w, y_f) :- at_idx(w, 10), at_idx(g, 21).
onface(wg_c, g, o_f) :- at_idx(w, 10), at_idx(g, 21).
onface(wg_c, g, y_f) :- at_idx(g, 10), at_idx(w, 21).
onface(wg_c, w, o_f) :- at_idx(g, 10), at_idx(w, 21).
onface(wb_c, w, y_f) :- at_idx(w, 10), at_idx(b, 21).
onface(wb_c, b, o_f) :- at_idx(w, 10), at_idx(b, 21).
onface(wb_c, b, y_f) :- at_idx(b, 10), at_idx(w, 21).
onface(wb_c, w, o_f) :- at_idx(b, 10), at_idx(w, 21).
onface(wr_c, w, y_f) :- at_idx(w, 10), at_idx(r, 21).
onface(wr_c, r, o_f) :- at_idx(w, 10), at_idx(r, 21).
onface(wr_c, r, y_f) :- at_idx(r, 10), at_idx(w, 21).
onface(wr_c, w, o_f) :- at_idx(r, 10), at_idx(w, 21).
onface(yo_c, y, y_f) :- at_idx(y, 10), at_idx(o, 21).
onface(yo_c, o, o_f) :- at_idx(y, 10), at_idx(o, 21).
onface(yo_c, o, y_f) :- at_idx(o, 10), at_idx(y, 21).
onface(yo_c, y, o_f) :- at_idx(o, 10), at_idx(y, 21).
onface(yb_c, y, y_f) :- at_idx(y, 10), at_idx(b, 21).
onface(yb_c, b, o_f) :- at_idx(y, 10), at_idx(b, 21).
onface(yb_c, b, y_f) :- at_idx(b, 10), at_idx(y, 21).
onface(yb_c, y, o_f) :- at_idx(b, 10), at_idx(y, 21).
onface(yg_c, y, y_f) :- at_idx(y, 10), at_idx(g, 21).
onface(yg_c, g, o_f) :- at_idx(y, 10), at_idx(g, 21).
onface(yg_c, g, y_f) :- at_idx(g, 10), at_idx(y, 21).
onface(yg_c, y, o_f) :- at_idx(g, 10), at_idx(y, 21).
onface(yr_c, y, y_f) :- at_idx(y, 10), at_idx(r, 21).
onface(yr_c, r, o_f) :- at_idx(y, 10), at_idx(r, 21).
onface(yr_c, r, y_f) :- at_idx(r, 10), at_idx(y, 21).
onface(yr_c, y, o_f) :- at_idx(r, 10), at_idx(y, 21).
onface(ob_c, o, y_f) :- at_idx(o, 10), at_idx(b, 21).
onface(ob_c, b, o_f) :- at_idx(o, 10), at_idx(b, 21).
onface(ob_c, b, y_f) :- at_idx(b, 10), at_idx(o, 21).
onface(ob_c, o, o_f) :- at_idx(b, 10), at_idx(o, 21).
onface(og_c, o, y_f) :- at_idx(o, 10), at_idx(g, 21).
onface(og_c, g, o_f) :- at_idx(o, 10), at_idx(g, 21).
onface(og_c, g, y_f) :- at_idx(g, 10), at_idx(o, 21).
onface(og_c, o, o_f) :- at_idx(g, 10), at_idx(o, 21).
onface(rg_c, r, y_f) :- at_idx(r, 10), at_idx(g, 21).
onface(rg_c, g, o_f) :- at_idx(r, 10), at_idx(g, 21).
onface(rg_c, g, y_f) :- at_idx(g, 10), at_idx(r, 21).
onface(rg_c, r, o_f) :- at_idx(g, 10), at_idx(r, 21).
onface(rb_c, r, y_f) :- at_idx(r, 10), at_idx(b, 21).
onface(rb_c, b, o_f) :- at_idx(r, 10), at_idx(b, 21).
onface(rb_c, b, y_f) :- at_idx(b, 10), at_idx(r, 21).
onface(rb_c, r, o_f) :- at_idx(b, 10), at_idx(r, 21).
onface(wo_c, w, y_f) :- at_idx(w, 12), at_idx(o, 39).
onface(wo_c, o, b_f) :- at_idx(w, 12), at_idx(o, 39).
onface(wo_c, o, y_f) :- at_idx(o, 12), at_idx(w, 39).
onface(wo_c, w, b_f) :- at_idx(o, 12), at_idx(w, 39).
onface(wg_c, w, y_f) :- at_idx(w, 12), at_idx(g, 39).
onface(wg_c, g, b_f) :- at_idx(w, 12), at_idx(g, 39).
onface(wg_c, g, y_f) :- at_idx(g, 12), at_idx(w, 39).
onface(wg_c, w, b_f) :- at_idx(g, 12), at_idx(w, 39).
onface(wb_c, w, y_f) :- at_idx(w, 12), at_idx(b, 39).
onface(wb_c, b, b_f) :- at_idx(w, 12), at_idx(b, 39).
onface(wb_c, b, y_f) :- at_idx(b, 12), at_idx(w, 39).
onface(wb_c, w, b_f) :- at_idx(b, 12), at_idx(w, 39).
onface(wr_c, w, y_f) :- at_idx(w, 12), at_idx(r, 39).
onface(wr_c, r, b_f) :- at_idx(w, 12), at_idx(r, 39).
onface(wr_c, r, y_f) :- at_idx(r, 12), at_idx(w, 39).
onface(wr_c, w, b_f) :- at_idx(r, 12), at_idx(w, 39).
onface(yo_c, y, y_f) :- at_idx(y, 12), at_idx(o, 39).
onface(yo_c, o, b_f) :- at_idx(y, 12), at_idx(o, 39).
onface(yo_c, o, y_f) :- at_idx(o, 12), at_idx(y, 39).
onface(yo_c, y, b_f) :- at_idx(o, 12), at_idx(y, 39).
onface(yb_c, y, y_f) :- at_idx(y, 12), at_idx(b, 39).
onface(yb_c, b, b_f) :- at_idx(y, 12), at_idx(b, 39).
onface(yb_c, b, y_f) :- at_idx(b, 12), at_idx(y, 39).
onface(yb_c, y, b_f) :- at_idx(b, 12), at_idx(y, 39).
onface(yg_c, y, y_f) :- at_idx(y, 12), at_idx(g, 39).
onface(yg_c, g, b_f) :- at_idx(y, 12), at_idx(g, 39).
onface(yg_c, g, y_f) :- at_idx(g, 12), at_idx(y, 39).
onface(yg_c, y, b_f) :- at_idx(g, 12), at_idx(y, 39).
onface(yr_c, y, y_f) :- at_idx(y, 12), at_idx(r, 39).
onface(yr_c, r, b_f) :- at_idx(y, 12), at_idx(r, 39).
onface(yr_c, r, y_f) :- at_idx(r, 12), at_idx(y, 39).
onface(yr_c, y, b_f) :- at_idx(r, 12), at_idx(y, 39).
onface(ob_c, o, y_f) :- at_idx(o, 12), at_idx(b, 39).
onface(ob_c, b, b_f) :- at_idx(o, 12), at_idx(b, 39).
onface(ob_c, b, y_f) :- at_idx(b, 12), at_idx(o, 39).
onface(ob_c, o, b_f) :- at_idx(b, 12), at_idx(o, 39).
onface(og_c, o, y_f) :- at_idx(o, 12), at_idx(g, 39).
onface(og_c, g, b_f) :- at_idx(o, 12), at_idx(g, 39).
onface(og_c, g, y_f) :- at_idx(g, 12), at_idx(o, 39).
onface(og_c, o, b_f) :- at_idx(g, 12), at_idx(o, 39).
onface(rg_c, r, y_f) :- at_idx(r, 12), at_idx(g, 39).
onface(rg_c, g, b_f) :- at_idx(r, 12), at_idx(g, 39).
onface(rg_c, g, y_f) :- at_idx(g, 12), at_idx(r, 39).
onface(rg_c, r, b_f) :- at_idx(g, 12), at_idx(r, 39).
onface(rb_c, r, y_f) :- at_idx(r, 12), at_idx(b, 39).
onface(rb_c, b, b_f) :- at_idx(r, 12), at_idx(b, 39).
onface(rb_c, b, y_f) :- at_idx(b, 12), at_idx(r, 39).
onface(rb_c, r, b_f) :- at_idx(b, 12), at_idx(r, 39).
onface(wo_c, w, y_f) :- at_idx(w, 14), at_idx(o, 48).
onface(wo_c, o, g_f) :- at_idx(w, 14), at_idx(o, 48).
onface(wo_c, o, y_f) :- at_idx(o, 14), at_idx(w, 48).
onface(wo_c, w, g_f) :- at_idx(o, 14), at_idx(w, 48).
onface(wg_c, w, y_f) :- at_idx(w, 14), at_idx(g, 48).
onface(wg_c, g, g_f) :- at_idx(w, 14), at_idx(g, 48).
onface(wg_c, g, y_f) :- at_idx(g, 14), at_idx(w, 48).
onface(wg_c, w, g_f) :- at_idx(g, 14), at_idx(w, 48).
onface(wb_c, w, y_f) :- at_idx(w, 14), at_idx(b, 48).
onface(wb_c, b, g_f) :- at_idx(w, 14), at_idx(b, 48).
onface(wb_c, b, y_f) :- at_idx(b, 14), at_idx(w, 48).
onface(wb_c, w, g_f) :- at_idx(b, 14), at_idx(w, 48).
onface(wr_c, w, y_f) :- at_idx(w, 14), at_idx(r, 48).
onface(wr_c, r, g_f) :- at_idx(w, 14), at_idx(r, 48).
onface(wr_c, r, y_f) :- at_idx(r, 14), at_idx(w, 48).
onface(wr_c, w, g_f) :- at_idx(r, 14), at_idx(w, 48).
onface(yo_c, y, y_f) :- at_idx(y, 14), at_idx(o, 48).
onface(yo_c, o, g_f) :- at_idx(y, 14), at_idx(o, 48).
onface(yo_c, o, y_f) :- at_idx(o, 14), at_idx(y, 48).
onface(yo_c, y, g_f) :- at_idx(o, 14), at_idx(y, 48).
onface(yb_c, y, y_f) :- at_idx(y, 14), at_idx(b, 48).
onface(yb_c, b, g_f) :- at_idx(y, 14), at_idx(b, 48).
onface(yb_c, b, y_f) :- at_idx(b, 14), at_idx(y, 48).
onface(yb_c, y, g_f) :- at_idx(b, 14), at_idx(y, 48).
onface(yg_c, y, y_f) :- at_idx(y, 14), at_idx(g, 48).
onface(yg_c, g, g_f) :- at_idx(y, 14), at_idx(g, 48).
onface(yg_c, g, y_f) :- at_idx(g, 14), at_idx(y, 48).
onface(yg_c, y, g_f) :- at_idx(g, 14), at_idx(y, 48).
onface(yr_c, y, y_f) :- at_idx(y, 14), at_idx(r, 48).
onface(yr_c, r, g_f) :- at_idx(y, 14), at_idx(r, 48).
onface(yr_c, r, y_f) :- at_idx(r, 14), at_idx(y, 48).
onface(yr_c, y, g_f) :- at_idx(r, 14), at_idx(y, 48).
onface(ob_c, o, y_f) :- at_idx(o, 14), at_idx(b, 48).
onface(ob_c, b, g_f) :- at_idx(o, 14), at_idx(b, 48).
onface(ob_c, b, y_f) :- at_idx(b, 14), at_idx(o, 48).
onface(ob_c, o, g_f) :- at_idx(b, 14), at_idx(o, 48).
onface(og_c, o, y_f) :- at_idx(o, 14), at_idx(g, 48).
onface(og_c, g, g_f) :- at_idx(o, 14), at_idx(g, 48).
onface(og_c, g, y_f) :- at_idx(g, 14), at_idx(o, 48).
onface(og_c, o, g_f) :- at_idx(g, 14), at_idx(o, 48).
onface(rg_c, r, y_f) :- at_idx(r, 14), at_idx(g, 48).
onface(rg_c, g, g_f) :- at_idx(r, 14), at_idx(g, 48).
onface(rg_c, g, y_f) :- at_idx(g, 14), at_idx(r, 48).
onface(rg_c, r, g_f) :- at_idx(g, 14), at_idx(r, 48).
onface(rb_c, r, y_f) :- at_idx(r, 14), at_idx(b, 48).
onface(rb_c, b, g_f) :- at_idx(r, 14), at_idx(b, 48).
onface(rb_c, b, y_f) :- at_idx(b, 14), at_idx(r, 48).
onface(rb_c, r, g_f) :- at_idx(b, 14), at_idx(r, 48).
onface(wo_c, w, y_f) :- at_idx(w, 16), at_idx(o, 30).
onface(wo_c, o, r_f) :- at_idx(w, 16), at_idx(o, 30).
onface(wo_c, o, y_f) :- at_idx(o, 16), at_idx(w, 30).
onface(wo_c, w, r_f) :- at_idx(o, 16), at_idx(w, 30).
onface(wg_c, w, y_f) :- at_idx(w, 16), at_idx(g, 30).
onface(wg_c, g, r_f) :- at_idx(w, 16), at_idx(g, 30).
onface(wg_c, g, y_f) :- at_idx(g, 16), at_idx(w, 30).
onface(wg_c, w, r_f) :- at_idx(g, 16), at_idx(w, 30).
onface(wb_c, w, y_f) :- at_idx(w, 16), at_idx(b, 30).
onface(wb_c, b, r_f) :- at_idx(w, 16), at_idx(b, 30).
onface(wb_c, b, y_f) :- at_idx(b, 16), at_idx(w, 30).
onface(wb_c, w, r_f) :- at_idx(b, 16), at_idx(w, 30).
onface(wr_c, w, y_f) :- at_idx(w, 16), at_idx(r, 30).
onface(wr_c, r, r_f) :- at_idx(w, 16), at_idx(r, 30).
onface(wr_c, r, y_f) :- at_idx(r, 16), at_idx(w, 30).
onface(wr_c, w, r_f) :- at_idx(r, 16), at_idx(w, 30).
onface(yo_c, y, y_f) :- at_idx(y, 16), at_idx(o, 30).
onface(yo_c, o, r_f) :- at_idx(y, 16), at_idx(o, 30).
onface(yo_c, o, y_f) :- at_idx(o, 16), at_idx(y, 30).
onface(yo_c, y, r_f) :- at_idx(o, 16), at_idx(y, 30).
onface(yb_c, y, y_f) :- at_idx(y, 16), at_idx(b, 30).
onface(yb_c, b, r_f) :- at_idx(y, 16), at_idx(b, 30).
onface(yb_c, b, y_f) :- at_idx(b, 16), at_idx(y, 30).
onface(yb_c, y, r_f) :- at_idx(b, 16), at_idx(y, 30).
onface(yg_c, y, y_f) :- at_idx(y, 16), at_idx(g, 30).
onface(yg_c, g, r_f) :- at_idx(y, 16), at_idx(g, 30).
onface(yg_c, g, y_f) :- at_idx(g, 16), at_idx(y, 30).
onface(yg_c, y, r_f) :- at_idx(g, 16), at_idx(y, 30).
onface(yr_c, y, y_f) :- at_idx(y, 16), at_idx(r, 30).
onface(yr_c, r, r_f) :- at_idx(y, 16), at_idx(r, 30).
onface(yr_c, r, y_f) :- at_idx(r, 16), at_idx(y, 30).
onface(yr_c, y, r_f) :- at_idx(r, 16), at_idx(y, 30).
onface(ob_c, o, y_f) :- at_idx(o, 16), at_idx(b, 30).
onface(ob_c, b, r_f) :- at_idx(o, 16), at_idx(b, 30).
onface(ob_c, b, y_f) :- at_idx(b, 16), at_idx(o, 30).
onface(ob_c, o, r_f) :- at_idx(b, 16), at_idx(o, 30).
onface(og_c, o, y_f) :- at_idx(o, 16), at_idx(g, 30).
onface(og_c, g, r_f) :- at_idx(o, 16), at_idx(g, 30).
onface(og_c, g, y_f) :- at_idx(g, 16), at_idx(o, 30).
onface(og_c, o, r_f) :- at_idx(g, 16), at_idx(o, 30).
onface(rg_c, r, y_f) :- at_idx(r, 16), at_idx(g, 30).
onface(rg_c, g, r_f) :- at_idx(r, 16), at_idx(g, 30).
onface(rg_c, g, y_f) :- at_idx(g, 16), at_idx(r, 30).
onface(rg_c, r, r_f) :- at_idx(g, 16), at_idx(r, 30).
onface(rb_c, r, y_f) :- at_idx(r, 16), at_idx(b, 30).
onface(rb_c, b, r_f) :- at_idx(r, 16), at_idx(b, 30).
onface(rb_c, b, y_f) :- at_idx(b, 16), at_idx(r, 30).
onface(rb_c, r, r_f) :- at_idx(b, 16), at_idx(r, 30).
onface(wo_c, w, o_f) :- at_idx(w, 19), at_idx(o, 43).
onface(wo_c, o, b_f) :- at_idx(w, 19), at_idx(o, 43).
onface(wo_c, o, o_f) :- at_idx(o, 19), at_idx(w, 43).
onface(wo_c, w, b_f) :- at_idx(o, 19), at_idx(w, 43).
onface(wg_c, w, o_f) :- at_idx(w, 19), at_idx(g, 43).
onface(wg_c, g, b_f) :- at_idx(w, 19), at_idx(g, 43).
onface(wg_c, g, o_f) :- at_idx(g, 19), at_idx(w, 43).
onface(wg_c, w, b_f) :- at_idx(g, 19), at_idx(w, 43).
onface(wb_c, w, o_f) :- at_idx(w, 19), at_idx(b, 43).
onface(wb_c, b, b_f) :- at_idx(w, 19), at_idx(b, 43).
onface(wb_c, b, o_f) :- at_idx(b, 19), at_idx(w, 43).
onface(wb_c, w, b_f) :- at_idx(b, 19), at_idx(w, 43).
onface(wr_c, w, o_f) :- at_idx(w, 19), at_idx(r, 43).
onface(wr_c, r, b_f) :- at_idx(w, 19), at_idx(r, 43).
onface(wr_c, r, o_f) :- at_idx(r, 19), at_idx(w, 43).
onface(wr_c, w, b_f) :- at_idx(r, 19), at_idx(w, 43).
onface(yo_c, y, o_f) :- at_idx(y, 19), at_idx(o, 43).
onface(yo_c, o, b_f) :- at_idx(y, 19), at_idx(o, 43).
onface(yo_c, o, o_f) :- at_idx(o, 19), at_idx(y, 43).
onface(yo_c, y, b_f) :- at_idx(o, 19), at_idx(y, 43).
onface(yb_c, y, o_f) :- at_idx(y, 19), at_idx(b, 43).
onface(yb_c, b, b_f) :- at_idx(y, 19), at_idx(b, 43).
onface(yb_c, b, o_f) :- at_idx(b, 19), at_idx(y, 43).
onface(yb_c, y, b_f) :- at_idx(b, 19), at_idx(y, 43).
onface(yg_c, y, o_f) :- at_idx(y, 19), at_idx(g, 43).
onface(yg_c, g, b_f) :- at_idx(y, 19), at_idx(g, 43).
onface(yg_c, g, o_f) :- at_idx(g, 19), at_idx(y, 43).
onface(yg_c, y, b_f) :- at_idx(g, 19), at_idx(y, 43).
onface(yr_c, y, o_f) :- at_idx(y, 19), at_idx(r, 43).
onface(yr_c, r, b_f) :- at_idx(y, 19), at_idx(r, 43).
onface(yr_c, r, o_f) :- at_idx(r, 19), at_idx(y, 43).
onface(yr_c, y, b_f) :- at_idx(r, 19), at_idx(y, 43).
onface(ob_c, o, o_f) :- at_idx(o, 19), at_idx(b, 43).
onface(ob_c, b, b_f) :- at_idx(o, 19), at_idx(b, 43).
onface(ob_c, b, o_f) :- at_idx(b, 19), at_idx(o, 43).
onface(ob_c, o, b_f) :- at_idx(b, 19), at_idx(o, 43).
onface(og_c, o, o_f) :- at_idx(o, 19), at_idx(g, 43).
onface(og_c, g, b_f) :- at_idx(o, 19), at_idx(g, 43).
onface(og_c, g, o_f) :- at_idx(g, 19), at_idx(o, 43).
onface(og_c, o, b_f) :- at_idx(g, 19), at_idx(o, 43).
onface(rg_c, r, o_f) :- at_idx(r, 19), at_idx(g, 43).
onface(rg_c, g, b_f) :- at_idx(r, 19), at_idx(g, 43).
onface(rg_c, g, o_f) :- at_idx(g, 19), at_idx(r, 43).
onface(rg_c, r, b_f) :- at_idx(g, 19), at_idx(r, 43).
onface(rb_c, r, o_f) :- at_idx(r, 19), at_idx(b, 43).
onface(rb_c, b, b_f) :- at_idx(r, 19), at_idx(b, 43).
onface(rb_c, b, o_f) :- at_idx(b, 19), at_idx(r, 43).
onface(rb_c, r, b_f) :- at_idx(b, 19), at_idx(r, 43).
onface(wo_c, w, o_f) :- at_idx(w, 25), at_idx(o, 46).
onface(wo_c, o, g_f) :- at_idx(w, 25), at_idx(o, 46).
onface(wo_c, o, o_f) :- at_idx(o, 25), at_idx(w, 46).
onface(wo_c, w, g_f) :- at_idx(o, 25), at_idx(w, 46).
onface(wg_c, w, o_f) :- at_idx(w, 25), at_idx(g, 46).
onface(wg_c, g, g_f) :- at_idx(w, 25), at_idx(g, 46).
onface(wg_c, g, o_f) :- at_idx(g, 25), at_idx(w, 46).
onface(wg_c, w, g_f) :- at_idx(g, 25), at_idx(w, 46).
onface(wb_c, w, o_f) :- at_idx(w, 25), at_idx(b, 46).
onface(wb_c, b, g_f) :- at_idx(w, 25), at_idx(b, 46).
onface(wb_c, b, o_f) :- at_idx(b, 25), at_idx(w, 46).
onface(wb_c, w, g_f) :- at_idx(b, 25), at_idx(w, 46).
onface(wr_c, w, o_f) :- at_idx(w, 25), at_idx(r, 46).
onface(wr_c, r, g_f) :- at_idx(w, 25), at_idx(r, 46).
onface(wr_c, r, o_f) :- at_idx(r, 25), at_idx(w, 46).
onface(wr_c, w, g_f) :- at_idx(r, 25), at_idx(w, 46).
onface(yo_c, y, o_f) :- at_idx(y, 25), at_idx(o, 46).
onface(yo_c, o, g_f) :- at_idx(y, 25), at_idx(o, 46).
onface(yo_c, o, o_f) :- at_idx(o, 25), at_idx(y, 46).
onface(yo_c, y, g_f) :- at_idx(o, 25), at_idx(y, 46).
onface(yb_c, y, o_f) :- at_idx(y, 25), at_idx(b, 46).
onface(yb_c, b, g_f) :- at_idx(y, 25), at_idx(b, 46).
onface(yb_c, b, o_f) :- at_idx(b, 25), at_idx(y, 46).
onface(yb_c, y, g_f) :- at_idx(b, 25), at_idx(y, 46).
onface(yg_c, y, o_f) :- at_idx(y, 25), at_idx(g, 46).
onface(yg_c, g, g_f) :- at_idx(y, 25), at_idx(g, 46).
onface(yg_c, g, o_f) :- at_idx(g, 25), at_idx(y, 46).
onface(yg_c, y, g_f) :- at_idx(g, 25), at_idx(y, 46).
onface(yr_c, y, o_f) :- at_idx(y, 25), at_idx(r, 46).
onface(yr_c, r, g_f) :- at_idx(y, 25), at_idx(r, 46).
onface(yr_c, r, o_f) :- at_idx(r, 25), at_idx(y, 46).
onface(yr_c, y, g_f) :- at_idx(r, 25), at_idx(y, 46).
onface(ob_c, o, o_f) :- at_idx(o, 25), at_idx(b, 46).
onface(ob_c, b, g_f) :- at_idx(o, 25), at_idx(b, 46).
onface(ob_c, b, o_f) :- at_idx(b, 25), at_idx(o, 46).
onface(ob_c, o, g_f) :- at_idx(b, 25), at_idx(o, 46).
onface(og_c, o, o_f) :- at_idx(o, 25), at_idx(g, 46).
onface(og_c, g, g_f) :- at_idx(o, 25), at_idx(g, 46).
onface(og_c, g, o_f) :- at_idx(g, 25), at_idx(o, 46).
onface(og_c, o, g_f) :- at_idx(g, 25), at_idx(o, 46).
onface(rg_c, r, o_f) :- at_idx(r, 25), at_idx(g, 46).
onface(rg_c, g, g_f) :- at_idx(r, 25), at_idx(g, 46).
onface(rg_c, g, o_f) :- at_idx(g, 25), at_idx(r, 46).
onface(rg_c, r, g_f) :- at_idx(g, 25), at_idx(r, 46).
onface(rb_c, r, o_f) :- at_idx(r, 25), at_idx(b, 46).
onface(rb_c, b, g_f) :- at_idx(r, 25), at_idx(b, 46).
onface(rb_c, b, o_f) :- at_idx(b, 25), at_idx(r, 46).
onface(rb_c, r, g_f) :- at_idx(b, 25), at_idx(r, 46).
onface(wo_c, w, r_f) :- at_idx(w, 28), at_idx(o, 52).
onface(wo_c, o, g_f) :- at_idx(w, 28), at_idx(o, 52).
onface(wo_c, o, r_f) :- at_idx(o, 28), at_idx(w, 52).
onface(wo_c, w, g_f) :- at_idx(o, 28), at_idx(w, 52).
onface(wg_c, w, r_f) :- at_idx(w, 28), at_idx(g, 52).
onface(wg_c, g, g_f) :- at_idx(w, 28), at_idx(g, 52).
onface(wg_c, g, r_f) :- at_idx(g, 28), at_idx(w, 52).
onface(wg_c, w, g_f) :- at_idx(g, 28), at_idx(w, 52).
onface(wb_c, w, r_f) :- at_idx(w, 28), at_idx(b, 52).
onface(wb_c, b, g_f) :- at_idx(w, 28), at_idx(b, 52).
onface(wb_c, b, r_f) :- at_idx(b, 28), at_idx(w, 52).
onface(wb_c, w, g_f) :- at_idx(b, 28), at_idx(w, 52).
onface(wr_c, w, r_f) :- at_idx(w, 28), at_idx(r, 52).
onface(wr_c, r, g_f) :- at_idx(w, 28), at_idx(r, 52).
onface(wr_c, r, r_f) :- at_idx(r, 28), at_idx(w, 52).
onface(wr_c, w, g_f) :- at_idx(r, 28), at_idx(w, 52).
onface(yo_c, y, r_f) :- at_idx(y, 28), at_idx(o, 52).
onface(yo_c, o, g_f) :- at_idx(y, 28), at_idx(o, 52).
onface(yo_c, o, r_f) :- at_idx(o, 28), at_idx(y, 52).
onface(yo_c, y, g_f) :- at_idx(o, 28), at_idx(y, 52).
onface(yb_c, y, r_f) :- at_idx(y, 28), at_idx(b, 52).
onface(yb_c, b, g_f) :- at_idx(y, 28), at_idx(b, 52).
onface(yb_c, b, r_f) :- at_idx(b, 28), at_idx(y, 52).
onface(yb_c, y, g_f) :- at_idx(b, 28), at_idx(y, 52).
onface(yg_c, y, r_f) :- at_idx(y, 28), at_idx(g, 52).
onface(yg_c, g, g_f) :- at_idx(y, 28), at_idx(g, 52).
onface(yg_c, g, r_f) :- at_idx(g, 28), at_idx(y, 52).
onface(yg_c, y, g_f) :- at_idx(g, 28), at_idx(y, 52).
onface(yr_c, y, r_f) :- at_idx(y, 28), at_idx(r, 52).
onface(yr_c, r, g_f) :- at_idx(y, 28), at_idx(r, 52).
onface(yr_c, r, r_f) :- at_idx(r, 28), at_idx(y, 52).
onface(yr_c, y, g_f) :- at_idx(r, 28), at_idx(y, 52).
onface(ob_c, o, r_f) :- at_idx(o, 28), at_idx(b, 52).
onface(ob_c, b, g_f) :- at_idx(o, 28), at_idx(b, 52).
onface(ob_c, b, r_f) :- at_idx(b, 28), at_idx(o, 52).
onface(ob_c, o, g_f) :- at_idx(b, 28), at_idx(o, 52).
onface(og_c, o, r_f) :- at_idx(o, 28), at_idx(g, 52).
onface(og_c, g, g_f) :- at_idx(o, 28), at_idx(g, 52).
onface(og_c, g, r_f) :- at_idx(g, 28), at_idx(o, 52).
onface(og_c, o, g_f) :- at_idx(g, 28), at_idx(o, 52).
onface(rg_c, r, r_f) :- at_idx(r, 28), at_idx(g, 52).
onface(rg_c, g, g_f) :- at_idx(r, 28), at_idx(g, 52).
onface(rg_c, g, r_f) :- at_idx(g, 28), at_idx(r, 52).
onface(rg_c, r, g_f) :- at_idx(g, 28), at_idx(r, 52).
onface(rb_c, r, r_f) :- at_idx(r, 28), at_idx(b, 52).
onface(rb_c, b, g_f) :- at_idx(r, 28), at_idx(b, 52).
onface(rb_c, b, r_f) :- at_idx(b, 28), at_idx(r, 52).
onface(rb_c, r, g_f) :- at_idx(b, 28), at_idx(r, 52).
onface(wo_c, w, r_f) :- at_idx(w, 34), at_idx(o, 37).
onface(wo_c, o, b_f) :- at_idx(w, 34), at_idx(o, 37).
onface(wo_c, o, r_f) :- at_idx(o, 34), at_idx(w, 37).
onface(wo_c, w, b_f) :- at_idx(o, 34), at_idx(w, 37).
onface(wg_c, w, r_f) :- at_idx(w, 34), at_idx(g, 37).
onface(wg_c, g, b_f) :- at_idx(w, 34), at_idx(g, 37).
onface(wg_c, g, r_f) :- at_idx(g, 34), at_idx(w, 37).
onface(wg_c, w, b_f) :- at_idx(g, 34), at_idx(w, 37).
onface(wb_c, w, r_f) :- at_idx(w, 34), at_idx(b, 37).
onface(wb_c, b, b_f) :- at_idx(w, 34), at_idx(b, 37).
onface(wb_c, b, r_f) :- at_idx(b, 34), at_idx(w, 37).
onface(wb_c, w, b_f) :- at_idx(b, 34), at_idx(w, 37).
onface(wr_c, w, r_f) :- at_idx(w, 34), at_idx(r, 37).
onface(wr_c, r, b_f) :- at_idx(w, 34), at_idx(r, 37).
onface(wr_c, r, r_f) :- at_idx(r, 34), at_idx(w, 37).
onface(wr_c, w, b_f) :- at_idx(r, 34), at_idx(w, 37).
onface(yo_c, y, r_f) :- at_idx(y, 34), at_idx(o, 37).
onface(yo_c, o, b_f) :- at_idx(y, 34), at_idx(o, 37).
onface(yo_c, o, r_f) :- at_idx(o, 34), at_idx(y, 37).
onface(yo_c, y, b_f) :- at_idx(o, 34), at_idx(y, 37).
onface(yb_c, y, r_f) :- at_idx(y, 34), at_idx(b, 37).
onface(yb_c, b, b_f) :- at_idx(y, 34), at_idx(b, 37).
onface(yb_c, b, r_f) :- at_idx(b, 34), at_idx(y, 37).
onface(yb_c, y, b_f) :- at_idx(b, 34), at_idx(y, 37).
onface(yg_c, y, r_f) :- at_idx(y, 34), at_idx(g, 37).
onface(yg_c, g, b_f) :- at_idx(y, 34), at_idx(g, 37).
onface(yg_c, g, r_f) :- at_idx(g, 34), at_idx(y, 37).
onface(yg_c, y, b_f) :- at_idx(g, 34), at_idx(y, 37).
onface(yr_c, y, r_f) :- at_idx(y, 34), at_idx(r, 37).
onface(yr_c, r, b_f) :- at_idx(y, 34), at_idx(r, 37).
onface(yr_c, r, r_f) :- at_idx(r, 34), at_idx(y, 37).
onface(yr_c, y, b_f) :- at_idx(r, 34), at_idx(y, 37).
onface(ob_c, o, r_f) :- at_idx(o, 34), at_idx(b, 37).
onface(ob_c, b, b_f) :- at_idx(o, 34), at_idx(b, 37).
onface(ob_c, b, r_f) :- at_idx(b, 34), at_idx(o, 37).
onface(ob_c, o, b_f) :- at_idx(b, 34), at_idx(o, 37).
onface(og_c, o, r_f) :- at_idx(o, 34), at_idx(g, 37).
onface(og_c, g, b_f) :- at_idx(o, 34), at_idx(g, 37).
onface(og_c, g, r_f) :- at_idx(g, 34), at_idx(o, 37).
onface(og_c, o, b_f) :- at_idx(g, 34), at_idx(o, 37).
onface(rg_c, r, r_f) :- at_idx(r, 34), at_idx(g, 37).
onface(rg_c, g, b_f) :- at_idx(r, 34), at_idx(g, 37).
onface(rg_c, g, r_f) :- at_idx(g, 34), at_idx(r, 37).
onface(rg_c, r, b_f) :- at_idx(g, 34), at_idx(r, 37).
onface(rb_c, r, r_f) :- at_idx(r, 34), at_idx(b, 37).
onface(rb_c, b, b_f) :- at_idx(r, 34), at_idx(b, 37).
onface(rb_c, b, r_f) :- at_idx(b, 34), at_idx(r, 37).
onface(rb_c, r, b_f) :- at_idx(b, 34), at_idx(r, 37).
onface(wog_c, w, w_f) :- at_idx(w, 0), at_idx(o, 26), at_idx(g, 47).
onface(wog_c, o, o_f) :- at_idx(w, 0), at_idx(o, 26), at_idx(g, 47).
onface(wog_c, g, g_f) :- at_idx(w, 0), at_idx(o, 26), at_idx(g, 47).
onface(wog_c, w, w_f) :- at_idx(w, 0), at_idx(g, 26), at_idx(o, 47).
onface(wog_c, g, o_f) :- at_idx(w, 0), at_idx(g, 26), at_idx(o, 47).
onface(wog_c, o, g_f) :- at_idx(w, 0), at_idx(g, 26), at_idx(o, 47).
onface(wog_c, o, w_f) :- at_idx(o, 0), at_idx(w, 26), at_idx(g, 47).
onface(wog_c, w, o_f) :- at_idx(o, 0), at_idx(w, 26), at_idx(g, 47).
onface(wog_c, g, g_f) :- at_idx(o, 0), at_idx(w, 26), at_idx(g, 47).
onface(wog_c, o, w_f) :- at_idx(o, 0), at_idx(g, 26), at_idx(w, 47).
onface(wog_c, g, o_f) :- at_idx(o, 0), at_idx(g, 26), at_idx(w, 47).
onface(wog_c, w, g_f) :- at_idx(o, 0), at_idx(g, 26), at_idx(w, 47).
onface(wog_c, g, w_f) :- at_idx(g, 0), at_idx(w, 26), at_idx(o, 47).
onface(wog_c, w, o_f) :- at_idx(g, 0), at_idx(w, 26), at_idx(o, 47).
onface(wog_c, o, g_f) :- at_idx(g, 0), at_idx(w, 26), at_idx(o, 47).
onface(wog_c, g, w_f) :- at_idx(g, 0), at_idx(o, 26), at_idx(w, 47).
onface(wog_c, o, o_f) :- at_idx(g, 0), at_idx(o, 26), at_idx(w, 47).
onface(wog_c, w, g_f) :- at_idx(g, 0), at_idx(o, 26), at_idx(w, 47).
onface(wob_c, w, w_f) :- at_idx(w, 0), at_idx(o, 26), at_idx(b, 47).
onface(wob_c, o, o_f) :- at_idx(w, 0), at_idx(o, 26), at_idx(b, 47).
onface(wob_c, b, g_f) :- at_idx(w, 0), at_idx(o, 26), at_idx(b, 47).
onface(wob_c, w, w_f) :- at_idx(w, 0), at_idx(b, 26), at_idx(o, 47).
onface(wob_c, b, o_f) :- at_idx(w, 0), at_idx(b, 26), at_idx(o, 47).
onface(wob_c, o, g_f) :- at_idx(w, 0), at_idx(b, 26), at_idx(o, 47).
onface(wob_c, o, w_f) :- at_idx(o, 0), at_idx(w, 26), at_idx(b, 47).
onface(wob_c, w, o_f) :- at_idx(o, 0), at_idx(w, 26), at_idx(b, 47).
onface(wob_c, b, g_f) :- at_idx(o, 0), at_idx(w, 26), at_idx(b, 47).
onface(wob_c, o, w_f) :- at_idx(o, 0), at_idx(b, 26), at_idx(w, 47).
onface(wob_c, b, o_f) :- at_idx(o, 0), at_idx(b, 26), at_idx(w, 47).
onface(wob_c, w, g_f) :- at_idx(o, 0), at_idx(b, 26), at_idx(w, 47).
onface(wob_c, b, w_f) :- at_idx(b, 0), at_idx(w, 26), at_idx(o, 47).
onface(wob_c, w, o_f) :- at_idx(b, 0), at_idx(w, 26), at_idx(o, 47).
onface(wob_c, o, g_f) :- at_idx(b, 0), at_idx(w, 26), at_idx(o, 47).
onface(wob_c, b, w_f) :- at_idx(b, 0), at_idx(o, 26), at_idx(w, 47).
onface(wob_c, o, o_f) :- at_idx(b, 0), at_idx(o, 26), at_idx(w, 47).
onface(wob_c, w, g_f) :- at_idx(b, 0), at_idx(o, 26), at_idx(w, 47).
onface(wrg_c, w, w_f) :- at_idx(w, 0), at_idx(r, 26), at_idx(g, 47).
onface(wrg_c, r, o_f) :- at_idx(w, 0), at_idx(r, 26), at_idx(g, 47).
onface(wrg_c, g, g_f) :- at_idx(w, 0), at_idx(r, 26), at_idx(g, 47).
onface(wrg_c, w, w_f) :- at_idx(w, 0), at_idx(g, 26), at_idx(r, 47).
onface(wrg_c, g, o_f) :- at_idx(w, 0), at_idx(g, 26), at_idx(r, 47).
onface(wrg_c, r, g_f) :- at_idx(w, 0), at_idx(g, 26), at_idx(r, 47).
onface(wrg_c, r, w_f) :- at_idx(r, 0), at_idx(w, 26), at_idx(g, 47).
onface(wrg_c, w, o_f) :- at_idx(r, 0), at_idx(w, 26), at_idx(g, 47).
onface(wrg_c, g, g_f) :- at_idx(r, 0), at_idx(w, 26), at_idx(g, 47).
onface(wrg_c, r, w_f) :- at_idx(r, 0), at_idx(g, 26), at_idx(w, 47).
onface(wrg_c, g, o_f) :- at_idx(r, 0), at_idx(g, 26), at_idx(w, 47).
onface(wrg_c, w, g_f) :- at_idx(r, 0), at_idx(g, 26), at_idx(w, 47).
onface(wrg_c, g, w_f) :- at_idx(g, 0), at_idx(w, 26), at_idx(r, 47).
onface(wrg_c, w, o_f) :- at_idx(g, 0), at_idx(w, 26), at_idx(r, 47).
onface(wrg_c, r, g_f) :- at_idx(g, 0), at_idx(w, 26), at_idx(r, 47).
onface(wrg_c, g, w_f) :- at_idx(g, 0), at_idx(r, 26), at_idx(w, 47).
onface(wrg_c, r, o_f) :- at_idx(g, 0), at_idx(r, 26), at_idx(w, 47).
onface(wrg_c, w, g_f) :- at_idx(g, 0), at_idx(r, 26), at_idx(w, 47).
onface(wrb_c, w, w_f) :- at_idx(w, 0), at_idx(r, 26), at_idx(b, 47).
onface(wrb_c, r, o_f) :- at_idx(w, 0), at_idx(r, 26), at_idx(b, 47).
onface(wrb_c, b, g_f) :- at_idx(w, 0), at_idx(r, 26), at_idx(b, 47).
onface(wrb_c, w, w_f) :- at_idx(w, 0), at_idx(b, 26), at_idx(r, 47).
onface(wrb_c, b, o_f) :- at_idx(w, 0), at_idx(b, 26), at_idx(r, 47).
onface(wrb_c, r, g_f) :- at_idx(w, 0), at_idx(b, 26), at_idx(r, 47).
onface(wrb_c, r, w_f) :- at_idx(r, 0), at_idx(w, 26), at_idx(b, 47).
onface(wrb_c, w, o_f) :- at_idx(r, 0), at_idx(w, 26), at_idx(b, 47).
onface(wrb_c, b, g_f) :- at_idx(r, 0), at_idx(w, 26), at_idx(b, 47).
onface(wrb_c, r, w_f) :- at_idx(r, 0), at_idx(b, 26), at_idx(w, 47).
onface(wrb_c, b, o_f) :- at_idx(r, 0), at_idx(b, 26), at_idx(w, 47).
onface(wrb_c, w, g_f) :- at_idx(r, 0), at_idx(b, 26), at_idx(w, 47).
onface(wrb_c, b, w_f) :- at_idx(b, 0), at_idx(w, 26), at_idx(r, 47).
onface(wrb_c, w, o_f) :- at_idx(b, 0), at_idx(w, 26), at_idx(r, 47).
onface(wrb_c, r, g_f) :- at_idx(b, 0), at_idx(w, 26), at_idx(r, 47).
onface(wrb_c, b, w_f) :- at_idx(b, 0), at_idx(r, 26), at_idx(w, 47).
onface(wrb_c, r, o_f) :- at_idx(b, 0), at_idx(r, 26), at_idx(w, 47).
onface(wrb_c, w, g_f) :- at_idx(b, 0), at_idx(r, 26), at_idx(w, 47).
onface(yob_c, y, w_f) :- at_idx(y, 0), at_idx(o, 26), at_idx(b, 47).
onface(yob_c, o, o_f) :- at_idx(y, 0), at_idx(o, 26), at_idx(b, 47).
onface(yob_c, b, g_f) :- at_idx(y, 0), at_idx(o, 26), at_idx(b, 47).
onface(yob_c, y, w_f) :- at_idx(y, 0), at_idx(b, 26), at_idx(o, 47).
onface(yob_c, b, o_f) :- at_idx(y, 0), at_idx(b, 26), at_idx(o, 47).
onface(yob_c, o, g_f) :- at_idx(y, 0), at_idx(b, 26), at_idx(o, 47).
onface(yob_c, o, w_f) :- at_idx(o, 0), at_idx(y, 26), at_idx(b, 47).
onface(yob_c, y, o_f) :- at_idx(o, 0), at_idx(y, 26), at_idx(b, 47).
onface(yob_c, b, g_f) :- at_idx(o, 0), at_idx(y, 26), at_idx(b, 47).
onface(yob_c, o, w_f) :- at_idx(o, 0), at_idx(b, 26), at_idx(y, 47).
onface(yob_c, b, o_f) :- at_idx(o, 0), at_idx(b, 26), at_idx(y, 47).
onface(yob_c, y, g_f) :- at_idx(o, 0), at_idx(b, 26), at_idx(y, 47).
onface(yob_c, b, w_f) :- at_idx(b, 0), at_idx(y, 26), at_idx(o, 47).
onface(yob_c, y, o_f) :- at_idx(b, 0), at_idx(y, 26), at_idx(o, 47).
onface(yob_c, o, g_f) :- at_idx(b, 0), at_idx(y, 26), at_idx(o, 47).
onface(yob_c, b, w_f) :- at_idx(b, 0), at_idx(o, 26), at_idx(y, 47).
onface(yob_c, o, o_f) :- at_idx(b, 0), at_idx(o, 26), at_idx(y, 47).
onface(yob_c, y, g_f) :- at_idx(b, 0), at_idx(o, 26), at_idx(y, 47).
onface(yog_c, y, w_f) :- at_idx(y, 0), at_idx(o, 26), at_idx(g, 47).
onface(yog_c, o, o_f) :- at_idx(y, 0), at_idx(o, 26), at_idx(g, 47).
onface(yog_c, g, g_f) :- at_idx(y, 0), at_idx(o, 26), at_idx(g, 47).
onface(yog_c, y, w_f) :- at_idx(y, 0), at_idx(g, 26), at_idx(o, 47).
onface(yog_c, g, o_f) :- at_idx(y, 0), at_idx(g, 26), at_idx(o, 47).
onface(yog_c, o, g_f) :- at_idx(y, 0), at_idx(g, 26), at_idx(o, 47).
onface(yog_c, o, w_f) :- at_idx(o, 0), at_idx(y, 26), at_idx(g, 47).
onface(yog_c, y, o_f) :- at_idx(o, 0), at_idx(y, 26), at_idx(g, 47).
onface(yog_c, g, g_f) :- at_idx(o, 0), at_idx(y, 26), at_idx(g, 47).
onface(yog_c, o, w_f) :- at_idx(o, 0), at_idx(g, 26), at_idx(y, 47).
onface(yog_c, g, o_f) :- at_idx(o, 0), at_idx(g, 26), at_idx(y, 47).
onface(yog_c, y, g_f) :- at_idx(o, 0), at_idx(g, 26), at_idx(y, 47).
onface(yog_c, g, w_f) :- at_idx(g, 0), at_idx(y, 26), at_idx(o, 47).
onface(yog_c, y, o_f) :- at_idx(g, 0), at_idx(y, 26), at_idx(o, 47).
onface(yog_c, o, g_f) :- at_idx(g, 0), at_idx(y, 26), at_idx(o, 47).
onface(yog_c, g, w_f) :- at_idx(g, 0), at_idx(o, 26), at_idx(y, 47).
onface(yog_c, o, o_f) :- at_idx(g, 0), at_idx(o, 26), at_idx(y, 47).
onface(yog_c, y, g_f) :- at_idx(g, 0), at_idx(o, 26), at_idx(y, 47).
onface(yrb_c, y, w_f) :- at_idx(y, 0), at_idx(r, 26), at_idx(b, 47).
onface(yrb_c, r, o_f) :- at_idx(y, 0), at_idx(r, 26), at_idx(b, 47).
onface(yrb_c, b, g_f) :- at_idx(y, 0), at_idx(r, 26), at_idx(b, 47).
onface(yrb_c, y, w_f) :- at_idx(y, 0), at_idx(b, 26), at_idx(r, 47).
onface(yrb_c, b, o_f) :- at_idx(y, 0), at_idx(b, 26), at_idx(r, 47).
onface(yrb_c, r, g_f) :- at_idx(y, 0), at_idx(b, 26), at_idx(r, 47).
onface(yrb_c, r, w_f) :- at_idx(r, 0), at_idx(y, 26), at_idx(b, 47).
onface(yrb_c, y, o_f) :- at_idx(r, 0), at_idx(y, 26), at_idx(b, 47).
onface(yrb_c, b, g_f) :- at_idx(r, 0), at_idx(y, 26), at_idx(b, 47).
onface(yrb_c, r, w_f) :- at_idx(r, 0), at_idx(b, 26), at_idx(y, 47).
onface(yrb_c, b, o_f) :- at_idx(r, 0), at_idx(b, 26), at_idx(y, 47).
onface(yrb_c, y, g_f) :- at_idx(r, 0), at_idx(b, 26), at_idx(y, 47).
onface(yrb_c, b, w_f) :- at_idx(b, 0), at_idx(y, 26), at_idx(r, 47).
onface(yrb_c, y, o_f) :- at_idx(b, 0), at_idx(y, 26), at_idx(r, 47).
onface(yrb_c, r, g_f) :- at_idx(b, 0), at_idx(y, 26), at_idx(r, 47).
onface(yrb_c, b, w_f) :- at_idx(b, 0), at_idx(r, 26), at_idx(y, 47).
onface(yrb_c, r, o_f) :- at_idx(b, 0), at_idx(r, 26), at_idx(y, 47).
onface(yrb_c, y, g_f) :- at_idx(b, 0), at_idx(r, 26), at_idx(y, 47).
onface(yrg_c, y, w_f) :- at_idx(y, 0), at_idx(r, 26), at_idx(g, 47).
onface(yrg_c, r, o_f) :- at_idx(y, 0), at_idx(r, 26), at_idx(g, 47).
onface(yrg_c, g, g_f) :- at_idx(y, 0), at_idx(r, 26), at_idx(g, 47).
onface(yrg_c, y, w_f) :- at_idx(y, 0), at_idx(g, 26), at_idx(r, 47).
onface(yrg_c, g, o_f) :- at_idx(y, 0), at_idx(g, 26), at_idx(r, 47).
onface(yrg_c, r, g_f) :- at_idx(y, 0), at_idx(g, 26), at_idx(r, 47).
onface(yrg_c, r, w_f) :- at_idx(r, 0), at_idx(y, 26), at_idx(g, 47).
onface(yrg_c, y, o_f) :- at_idx(r, 0), at_idx(y, 26), at_idx(g, 47).
onface(yrg_c, g, g_f) :- at_idx(r, 0), at_idx(y, 26), at_idx(g, 47).
onface(yrg_c, r, w_f) :- at_idx(r, 0), at_idx(g, 26), at_idx(y, 47).
onface(yrg_c, g, o_f) :- at_idx(r, 0), at_idx(g, 26), at_idx(y, 47).
onface(yrg_c, y, g_f) :- at_idx(r, 0), at_idx(g, 26), at_idx(y, 47).
onface(yrg_c, g, w_f) :- at_idx(g, 0), at_idx(y, 26), at_idx(r, 47).
onface(yrg_c, y, o_f) :- at_idx(g, 0), at_idx(y, 26), at_idx(r, 47).
onface(yrg_c, r, g_f) :- at_idx(g, 0), at_idx(y, 26), at_idx(r, 47).
onface(yrg_c, g, w_f) :- at_idx(g, 0), at_idx(r, 26), at_idx(y, 47).
onface(yrg_c, r, o_f) :- at_idx(g, 0), at_idx(r, 26), at_idx(y, 47).
onface(yrg_c, y, g_f) :- at_idx(g, 0), at_idx(r, 26), at_idx(y, 47).
onface(wog_c, w, w_f) :- at_idx(w, 2), at_idx(o, 20), at_idx(g, 44).
onface(wog_c, o, o_f) :- at_idx(w, 2), at_idx(o, 20), at_idx(g, 44).
onface(wog_c, g, b_f) :- at_idx(w, 2), at_idx(o, 20), at_idx(g, 44).
onface(wog_c, w, w_f) :- at_idx(w, 2), at_idx(g, 20), at_idx(o, 44).
onface(wog_c, g, o_f) :- at_idx(w, 2), at_idx(g, 20), at_idx(o, 44).
onface(wog_c, o, b_f) :- at_idx(w, 2), at_idx(g, 20), at_idx(o, 44).
onface(wog_c, o, w_f) :- at_idx(o, 2), at_idx(w, 20), at_idx(g, 44).
onface(wog_c, w, o_f) :- at_idx(o, 2), at_idx(w, 20), at_idx(g, 44).
onface(wog_c, g, b_f) :- at_idx(o, 2), at_idx(w, 20), at_idx(g, 44).
onface(wog_c, o, w_f) :- at_idx(o, 2), at_idx(g, 20), at_idx(w, 44).
onface(wog_c, g, o_f) :- at_idx(o, 2), at_idx(g, 20), at_idx(w, 44).
onface(wog_c, w, b_f) :- at_idx(o, 2), at_idx(g, 20), at_idx(w, 44).
onface(wog_c, g, w_f) :- at_idx(g, 2), at_idx(w, 20), at_idx(o, 44).
onface(wog_c, w, o_f) :- at_idx(g, 2), at_idx(w, 20), at_idx(o, 44).
onface(wog_c, o, b_f) :- at_idx(g, 2), at_idx(w, 20), at_idx(o, 44).
onface(wog_c, g, w_f) :- at_idx(g, 2), at_idx(o, 20), at_idx(w, 44).
onface(wog_c, o, o_f) :- at_idx(g, 2), at_idx(o, 20), at_idx(w, 44).
onface(wog_c, w, b_f) :- at_idx(g, 2), at_idx(o, 20), at_idx(w, 44).
onface(wob_c, w, w_f) :- at_idx(w, 2), at_idx(o, 20), at_idx(b, 44).
onface(wob_c, o, o_f) :- at_idx(w, 2), at_idx(o, 20), at_idx(b, 44).
onface(wob_c, b, b_f) :- at_idx(w, 2), at_idx(o, 20), at_idx(b, 44).
onface(wob_c, w, w_f) :- at_idx(w, 2), at_idx(b, 20), at_idx(o, 44).
onface(wob_c, b, o_f) :- at_idx(w, 2), at_idx(b, 20), at_idx(o, 44).
onface(wob_c, o, b_f) :- at_idx(w, 2), at_idx(b, 20), at_idx(o, 44).
onface(wob_c, o, w_f) :- at_idx(o, 2), at_idx(w, 20), at_idx(b, 44).
onface(wob_c, w, o_f) :- at_idx(o, 2), at_idx(w, 20), at_idx(b, 44).
onface(wob_c, b, b_f) :- at_idx(o, 2), at_idx(w, 20), at_idx(b, 44).
onface(wob_c, o, w_f) :- at_idx(o, 2), at_idx(b, 20), at_idx(w, 44).
onface(wob_c, b, o_f) :- at_idx(o, 2), at_idx(b, 20), at_idx(w, 44).
onface(wob_c, w, b_f) :- at_idx(o, 2), at_idx(b, 20), at_idx(w, 44).
onface(wob_c, b, w_f) :- at_idx(b, 2), at_idx(w, 20), at_idx(o, 44).
onface(wob_c, w, o_f) :- at_idx(b, 2), at_idx(w, 20), at_idx(o, 44).
onface(wob_c, o, b_f) :- at_idx(b, 2), at_idx(w, 20), at_idx(o, 44).
onface(wob_c, b, w_f) :- at_idx(b, 2), at_idx(o, 20), at_idx(w, 44).
onface(wob_c, o, o_f) :- at_idx(b, 2), at_idx(o, 20), at_idx(w, 44).
onface(wob_c, w, b_f) :- at_idx(b, 2), at_idx(o, 20), at_idx(w, 44).
onface(wrg_c, w, w_f) :- at_idx(w, 2), at_idx(r, 20), at_idx(g, 44).
onface(wrg_c, r, o_f) :- at_idx(w, 2), at_idx(r, 20), at_idx(g, 44).
onface(wrg_c, g, b_f) :- at_idx(w, 2), at_idx(r, 20), at_idx(g, 44).
onface(wrg_c, w, w_f) :- at_idx(w, 2), at_idx(g, 20), at_idx(r, 44).
onface(wrg_c, g, o_f) :- at_idx(w, 2), at_idx(g, 20), at_idx(r, 44).
onface(wrg_c, r, b_f) :- at_idx(w, 2), at_idx(g, 20), at_idx(r, 44).
onface(wrg_c, r, w_f) :- at_idx(r, 2), at_idx(w, 20), at_idx(g, 44).
onface(wrg_c, w, o_f) :- at_idx(r, 2), at_idx(w, 20), at_idx(g, 44).
onface(wrg_c, g, b_f) :- at_idx(r, 2), at_idx(w, 20), at_idx(g, 44).
onface(wrg_c, r, w_f) :- at_idx(r, 2), at_idx(g, 20), at_idx(w, 44).
onface(wrg_c, g, o_f) :- at_idx(r, 2), at_idx(g, 20), at_idx(w, 44).
onface(wrg_c, w, b_f) :- at_idx(r, 2), at_idx(g, 20), at_idx(w, 44).
onface(wrg_c, g, w_f) :- at_idx(g, 2), at_idx(w, 20), at_idx(r, 44).
onface(wrg_c, w, o_f) :- at_idx(g, 2), at_idx(w, 20), at_idx(r, 44).
onface(wrg_c, r, b_f) :- at_idx(g, 2), at_idx(w, 20), at_idx(r, 44).
onface(wrg_c, g, w_f) :- at_idx(g, 2), at_idx(r, 20), at_idx(w, 44).
onface(wrg_c, r, o_f) :- at_idx(g, 2), at_idx(r, 20), at_idx(w, 44).
onface(wrg_c, w, b_f) :- at_idx(g, 2), at_idx(r, 20), at_idx(w, 44).
onface(wrb_c, w, w_f) :- at_idx(w, 2), at_idx(r, 20), at_idx(b, 44).
onface(wrb_c, r, o_f) :- at_idx(w, 2), at_idx(r, 20), at_idx(b, 44).
onface(wrb_c, b, b_f) :- at_idx(w, 2), at_idx(r, 20), at_idx(b, 44).
onface(wrb_c, w, w_f) :- at_idx(w, 2), at_idx(b, 20), at_idx(r, 44).
onface(wrb_c, b, o_f) :- at_idx(w, 2), at_idx(b, 20), at_idx(r, 44).
onface(wrb_c, r, b_f) :- at_idx(w, 2), at_idx(b, 20), at_idx(r, 44).
onface(wrb_c, r, w_f) :- at_idx(r, 2), at_idx(w, 20), at_idx(b, 44).
onface(wrb_c, w, o_f) :- at_idx(r, 2), at_idx(w, 20), at_idx(b, 44).
onface(wrb_c, b, b_f) :- at_idx(r, 2), at_idx(w, 20), at_idx(b, 44).
onface(wrb_c, r, w_f) :- at_idx(r, 2), at_idx(b, 20), at_idx(w, 44).
onface(wrb_c, b, o_f) :- at_idx(r, 2), at_idx(b, 20), at_idx(w, 44).
onface(wrb_c, w, b_f) :- at_idx(r, 2), at_idx(b, 20), at_idx(w, 44).
onface(wrb_c, b, w_f) :- at_idx(b, 2), at_idx(w, 20), at_idx(r, 44).
onface(wrb_c, w, o_f) :- at_idx(b, 2), at_idx(w, 20), at_idx(r, 44).
onface(wrb_c, r, b_f) :- at_idx(b, 2), at_idx(w, 20), at_idx(r, 44).
onface(wrb_c, b, w_f) :- at_idx(b, 2), at_idx(r, 20), at_idx(w, 44).
onface(wrb_c, r, o_f) :- at_idx(b, 2), at_idx(r, 20), at_idx(w, 44).
onface(wrb_c, w, b_f) :- at_idx(b, 2), at_idx(r, 20), at_idx(w, 44).
onface(yob_c, y, w_f) :- at_idx(y, 2), at_idx(o, 20), at_idx(b, 44).
onface(yob_c, o, o_f) :- at_idx(y, 2), at_idx(o, 20), at_idx(b, 44).
onface(yob_c, b, b_f) :- at_idx(y, 2), at_idx(o, 20), at_idx(b, 44).
onface(yob_c, y, w_f) :- at_idx(y, 2), at_idx(b, 20), at_idx(o, 44).
onface(yob_c, b, o_f) :- at_idx(y, 2), at_idx(b, 20), at_idx(o, 44).
onface(yob_c, o, b_f) :- at_idx(y, 2), at_idx(b, 20), at_idx(o, 44).
onface(yob_c, o, w_f) :- at_idx(o, 2), at_idx(y, 20), at_idx(b, 44).
onface(yob_c, y, o_f) :- at_idx(o, 2), at_idx(y, 20), at_idx(b, 44).
onface(yob_c, b, b_f) :- at_idx(o, 2), at_idx(y, 20), at_idx(b, 44).
onface(yob_c, o, w_f) :- at_idx(o, 2), at_idx(b, 20), at_idx(y, 44).
onface(yob_c, b, o_f) :- at_idx(o, 2), at_idx(b, 20), at_idx(y, 44).
onface(yob_c, y, b_f) :- at_idx(o, 2), at_idx(b, 20), at_idx(y, 44).
onface(yob_c, b, w_f) :- at_idx(b, 2), at_idx(y, 20), at_idx(o, 44).
onface(yob_c, y, o_f) :- at_idx(b, 2), at_idx(y, 20), at_idx(o, 44).
onface(yob_c, o, b_f) :- at_idx(b, 2), at_idx(y, 20), at_idx(o, 44).
onface(yob_c, b, w_f) :- at_idx(b, 2), at_idx(o, 20), at_idx(y, 44).
onface(yob_c, o, o_f) :- at_idx(b, 2), at_idx(o, 20), at_idx(y, 44).
onface(yob_c, y, b_f) :- at_idx(b, 2), at_idx(o, 20), at_idx(y, 44).
onface(yog_c, y, w_f) :- at_idx(y, 2), at_idx(o, 20), at_idx(g, 44).
onface(yog_c, o, o_f) :- at_idx(y, 2), at_idx(o, 20), at_idx(g, 44).
onface(yog_c, g, b_f) :- at_idx(y, 2), at_idx(o, 20), at_idx(g, 44).
onface(yog_c, y, w_f) :- at_idx(y, 2), at_idx(g, 20), at_idx(o, 44).
onface(yog_c, g, o_f) :- at_idx(y, 2), at_idx(g, 20), at_idx(o, 44).
onface(yog_c, o, b_f) :- at_idx(y, 2), at_idx(g, 20), at_idx(o, 44).
onface(yog_c, o, w_f) :- at_idx(o, 2), at_idx(y, 20), at_idx(g, 44).
onface(yog_c, y, o_f) :- at_idx(o, 2), at_idx(y, 20), at_idx(g, 44).
onface(yog_c, g, b_f) :- at_idx(o, 2), at_idx(y, 20), at_idx(g, 44).
onface(yog_c, o, w_f) :- at_idx(o, 2), at_idx(g, 20), at_idx(y, 44).
onface(yog_c, g, o_f) :- at_idx(o, 2), at_idx(g, 20), at_idx(y, 44).
onface(yog_c, y, b_f) :- at_idx(o, 2), at_idx(g, 20), at_idx(y, 44).
onface(yog_c, g, w_f) :- at_idx(g, 2), at_idx(y, 20), at_idx(o, 44).
onface(yog_c, y, o_f) :- at_idx(g, 2), at_idx(y, 20), at_idx(o, 44).
onface(yog_c, o, b_f) :- at_idx(g, 2), at_idx(y, 20), at_idx(o, 44).
onface(yog_c, g, w_f) :- at_idx(g, 2), at_idx(o, 20), at_idx(y, 44).
onface(yog_c, o, o_f) :- at_idx(g, 2), at_idx(o, 20), at_idx(y, 44).
onface(yog_c, y, b_f) :- at_idx(g, 2), at_idx(o, 20), at_idx(y, 44).
onface(yrb_c, y, w_f) :- at_idx(y, 2), at_idx(r, 20), at_idx(b, 44).
onface(yrb_c, r, o_f) :- at_idx(y, 2), at_idx(r, 20), at_idx(b, 44).
onface(yrb_c, b, b_f) :- at_idx(y, 2), at_idx(r, 20), at_idx(b, 44).
onface(yrb_c, y, w_f) :- at_idx(y, 2), at_idx(b, 20), at_idx(r, 44).
onface(yrb_c, b, o_f) :- at_idx(y, 2), at_idx(b, 20), at_idx(r, 44).
onface(yrb_c, r, b_f) :- at_idx(y, 2), at_idx(b, 20), at_idx(r, 44).
onface(yrb_c, r, w_f) :- at_idx(r, 2), at_idx(y, 20), at_idx(b, 44).
onface(yrb_c, y, o_f) :- at_idx(r, 2), at_idx(y, 20), at_idx(b, 44).
onface(yrb_c, b, b_f) :- at_idx(r, 2), at_idx(y, 20), at_idx(b, 44).
onface(yrb_c, r, w_f) :- at_idx(r, 2), at_idx(b, 20), at_idx(y, 44).
onface(yrb_c, b, o_f) :- at_idx(r, 2), at_idx(b, 20), at_idx(y, 44).
onface(yrb_c, y, b_f) :- at_idx(r, 2), at_idx(b, 20), at_idx(y, 44).
onface(yrb_c, b, w_f) :- at_idx(b, 2), at_idx(y, 20), at_idx(r, 44).
onface(yrb_c, y, o_f) :- at_idx(b, 2), at_idx(y, 20), at_idx(r, 44).
onface(yrb_c, r, b_f) :- at_idx(b, 2), at_idx(y, 20), at_idx(r, 44).
onface(yrb_c, b, w_f) :- at_idx(b, 2), at_idx(r, 20), at_idx(y, 44).
onface(yrb_c, r, o_f) :- at_idx(b, 2), at_idx(r, 20), at_idx(y, 44).
onface(yrb_c, y, b_f) :- at_idx(b, 2), at_idx(r, 20), at_idx(y, 44).
onface(yrg_c, y, w_f) :- at_idx(y, 2), at_idx(r, 20), at_idx(g, 44).
onface(yrg_c, r, o_f) :- at_idx(y, 2), at_idx(r, 20), at_idx(g, 44).
onface(yrg_c, g, b_f) :- at_idx(y, 2), at_idx(r, 20), at_idx(g, 44).
onface(yrg_c, y, w_f) :- at_idx(y, 2), at_idx(g, 20), at_idx(r, 44).
onface(yrg_c, g, o_f) :- at_idx(y, 2), at_idx(g, 20), at_idx(r, 44).
onface(yrg_c, r, b_f) :- at_idx(y, 2), at_idx(g, 20), at_idx(r, 44).
onface(yrg_c, r, w_f) :- at_idx(r, 2), at_idx(y, 20), at_idx(g, 44).
onface(yrg_c, y, o_f) :- at_idx(r, 2), at_idx(y, 20), at_idx(g, 44).
onface(yrg_c, g, b_f) :- at_idx(r, 2), at_idx(y, 20), at_idx(g, 44).
onface(yrg_c, r, w_f) :- at_idx(r, 2), at_idx(g, 20), at_idx(y, 44).
onface(yrg_c, g, o_f) :- at_idx(r, 2), at_idx(g, 20), at_idx(y, 44).
onface(yrg_c, y, b_f) :- at_idx(r, 2), at_idx(g, 20), at_idx(y, 44).
onface(yrg_c, g, w_f) :- at_idx(g, 2), at_idx(y, 20), at_idx(r, 44).
onface(yrg_c, y, o_f) :- at_idx(g, 2), at_idx(y, 20), at_idx(r, 44).
onface(yrg_c, r, b_f) :- at_idx(g, 2), at_idx(y, 20), at_idx(r, 44).
onface(yrg_c, g, w_f) :- at_idx(g, 2), at_idx(r, 20), at_idx(y, 44).
onface(yrg_c, r, o_f) :- at_idx(g, 2), at_idx(r, 20), at_idx(y, 44).
onface(yrg_c, y, b_f) :- at_idx(g, 2), at_idx(r, 20), at_idx(y, 44).
onface(wog_c, w, w_f) :- at_idx(w, 6), at_idx(o, 29), at_idx(g, 53).
onface(wog_c, o, r_f) :- at_idx(w, 6), at_idx(o, 29), at_idx(g, 53).
onface(wog_c, g, g_f) :- at_idx(w, 6), at_idx(o, 29), at_idx(g, 53).
onface(wog_c, w, w_f) :- at_idx(w, 6), at_idx(g, 29), at_idx(o, 53).
onface(wog_c, g, r_f) :- at_idx(w, 6), at_idx(g, 29), at_idx(o, 53).
onface(wog_c, o, g_f) :- at_idx(w, 6), at_idx(g, 29), at_idx(o, 53).
onface(wog_c, o, w_f) :- at_idx(o, 6), at_idx(w, 29), at_idx(g, 53).
onface(wog_c, w, r_f) :- at_idx(o, 6), at_idx(w, 29), at_idx(g, 53).
onface(wog_c, g, g_f) :- at_idx(o, 6), at_idx(w, 29), at_idx(g, 53).
onface(wog_c, o, w_f) :- at_idx(o, 6), at_idx(g, 29), at_idx(w, 53).
onface(wog_c, g, r_f) :- at_idx(o, 6), at_idx(g, 29), at_idx(w, 53).
onface(wog_c, w, g_f) :- at_idx(o, 6), at_idx(g, 29), at_idx(w, 53).
onface(wog_c, g, w_f) :- at_idx(g, 6), at_idx(w, 29), at_idx(o, 53).
onface(wog_c, w, r_f) :- at_idx(g, 6), at_idx(w, 29), at_idx(o, 53).
onface(wog_c, o, g_f) :- at_idx(g, 6), at_idx(w, 29), at_idx(o, 53).
onface(wog_c, g, w_f) :- at_idx(g, 6), at_idx(o, 29), at_idx(w, 53).
onface(wog_c, o, r_f) :- at_idx(g, 6), at_idx(o, 29), at_idx(w, 53).
onface(wog_c, w, g_f) :- at_idx(g, 6), at_idx(o, 29), at_idx(w, 53).
onface(wob_c, w, w_f) :- at_idx(w, 6), at_idx(o, 29), at_idx(b, 53).
onface(wob_c, o, r_f) :- at_idx(w, 6), at_idx(o, 29), at_idx(b, 53).
onface(wob_c, b, g_f) :- at_idx(w, 6), at_idx(o, 29), at_idx(b, 53).
onface(wob_c, w, w_f) :- at_idx(w, 6), at_idx(b, 29), at_idx(o, 53).
onface(wob_c, b, r_f) :- at_idx(w, 6), at_idx(b, 29), at_idx(o, 53).
onface(wob_c, o, g_f) :- at_idx(w, 6), at_idx(b, 29), at_idx(o, 53).
onface(wob_c, o, w_f) :- at_idx(o, 6), at_idx(w, 29), at_idx(b, 53).
onface(wob_c, w, r_f) :- at_idx(o, 6), at_idx(w, 29), at_idx(b, 53).
onface(wob_c, b, g_f) :- at_idx(o, 6), at_idx(w, 29), at_idx(b, 53).
onface(wob_c, o, w_f) :- at_idx(o, 6), at_idx(b, 29), at_idx(w, 53).
onface(wob_c, b, r_f) :- at_idx(o, 6), at_idx(b, 29), at_idx(w, 53).
onface(wob_c, w, g_f) :- at_idx(o, 6), at_idx(b, 29), at_idx(w, 53).
onface(wob_c, b, w_f) :- at_idx(b, 6), at_idx(w, 29), at_idx(o, 53).
onface(wob_c, w, r_f) :- at_idx(b, 6), at_idx(w, 29), at_idx(o, 53).
onface(wob_c, o, g_f) :- at_idx(b, 6), at_idx(w, 29), at_idx(o, 53).
onface(wob_c, b, w_f) :- at_idx(b, 6), at_idx(o, 29), at_idx(w, 53).
onface(wob_c, o, r_f) :- at_idx(b, 6), at_idx(o, 29), at_idx(w, 53).
onface(wob_c, w, g_f) :- at_idx(b, 6), at_idx(o, 29), at_idx(w, 53).
onface(wrg_c, w, w_f) :- at_idx(w, 6), at_idx(r, 29), at_idx(g, 53).
onface(wrg_c, r, r_f) :- at_idx(w, 6), at_idx(r, 29), at_idx(g, 53).
onface(wrg_c, g, g_f) :- at_idx(w, 6), at_idx(r, 29), at_idx(g, 53).
onface(wrg_c, w, w_f) :- at_idx(w, 6), at_idx(g, 29), at_idx(r, 53).
onface(wrg_c, g, r_f) :- at_idx(w, 6), at_idx(g, 29), at_idx(r, 53).
onface(wrg_c, r, g_f) :- at_idx(w, 6), at_idx(g, 29), at_idx(r, 53).
onface(wrg_c, r, w_f) :- at_idx(r, 6), at_idx(w, 29), at_idx(g, 53).
onface(wrg_c, w, r_f) :- at_idx(r, 6), at_idx(w, 29), at_idx(g, 53).
onface(wrg_c, g, g_f) :- at_idx(r, 6), at_idx(w, 29), at_idx(g, 53).
onface(wrg_c, r, w_f) :- at_idx(r, 6), at_idx(g, 29), at_idx(w, 53).
onface(wrg_c, g, r_f) :- at_idx(r, 6), at_idx(g, 29), at_idx(w, 53).
onface(wrg_c, w, g_f) :- at_idx(r, 6), at_idx(g, 29), at_idx(w, 53).
onface(wrg_c, g, w_f) :- at_idx(g, 6), at_idx(w, 29), at_idx(r, 53).
onface(wrg_c, w, r_f) :- at_idx(g, 6), at_idx(w, 29), at_idx(r, 53).
onface(wrg_c, r, g_f) :- at_idx(g, 6), at_idx(w, 29), at_idx(r, 53).
onface(wrg_c, g, w_f) :- at_idx(g, 6), at_idx(r, 29), at_idx(w, 53).
onface(wrg_c, r, r_f) :- at_idx(g, 6), at_idx(r, 29), at_idx(w, 53).
onface(wrg_c, w, g_f) :- at_idx(g, 6), at_idx(r, 29), at_idx(w, 53).
onface(wrb_c, w, w_f) :- at_idx(w, 6), at_idx(r, 29), at_idx(b, 53).
onface(wrb_c, r, r_f) :- at_idx(w, 6), at_idx(r, 29), at_idx(b, 53).
onface(wrb_c, b, g_f) :- at_idx(w, 6), at_idx(r, 29), at_idx(b, 53).
onface(wrb_c, w, w_f) :- at_idx(w, 6), at_idx(b, 29), at_idx(r, 53).
onface(wrb_c, b, r_f) :- at_idx(w, 6), at_idx(b, 29), at_idx(r, 53).
onface(wrb_c, r, g_f) :- at_idx(w, 6), at_idx(b, 29), at_idx(r, 53).
onface(wrb_c, r, w_f) :- at_idx(r, 6), at_idx(w, 29), at_idx(b, 53).
onface(wrb_c, w, r_f) :- at_idx(r, 6), at_idx(w, 29), at_idx(b, 53).
onface(wrb_c, b, g_f) :- at_idx(r, 6), at_idx(w, 29), at_idx(b, 53).
onface(wrb_c, r, w_f) :- at_idx(r, 6), at_idx(b, 29), at_idx(w, 53).
onface(wrb_c, b, r_f) :- at_idx(r, 6), at_idx(b, 29), at_idx(w, 53).
onface(wrb_c, w, g_f) :- at_idx(r, 6), at_idx(b, 29), at_idx(w, 53).
onface(wrb_c, b, w_f) :- at_idx(b, 6), at_idx(w, 29), at_idx(r, 53).
onface(wrb_c, w, r_f) :- at_idx(b, 6), at_idx(w, 29), at_idx(r, 53).
onface(wrb_c, r, g_f) :- at_idx(b, 6), at_idx(w, 29), at_idx(r, 53).
onface(wrb_c, b, w_f) :- at_idx(b, 6), at_idx(r, 29), at_idx(w, 53).
onface(wrb_c, r, r_f) :- at_idx(b, 6), at_idx(r, 29), at_idx(w, 53).
onface(wrb_c, w, g_f) :- at_idx(b, 6), at_idx(r, 29), at_idx(w, 53).
onface(yob_c, y, w_f) :- at_idx(y, 6), at_idx(o, 29), at_idx(b, 53).
onface(yob_c, o, r_f) :- at_idx(y, 6), at_idx(o, 29), at_idx(b, 53).
onface(yob_c, b, g_f) :- at_idx(y, 6), at_idx(o, 29), at_idx(b, 53).
onface(yob_c, y, w_f) :- at_idx(y, 6), at_idx(b, 29), at_idx(o, 53).
onface(yob_c, b, r_f) :- at_idx(y, 6), at_idx(b, 29), at_idx(o, 53).
onface(yob_c, o, g_f) :- at_idx(y, 6), at_idx(b, 29), at_idx(o, 53).
onface(yob_c, o, w_f) :- at_idx(o, 6), at_idx(y, 29), at_idx(b, 53).
onface(yob_c, y, r_f) :- at_idx(o, 6), at_idx(y, 29), at_idx(b, 53).
onface(yob_c, b, g_f) :- at_idx(o, 6), at_idx(y, 29), at_idx(b, 53).
onface(yob_c, o, w_f) :- at_idx(o, 6), at_idx(b, 29), at_idx(y, 53).
onface(yob_c, b, r_f) :- at_idx(o, 6), at_idx(b, 29), at_idx(y, 53).
onface(yob_c, y, g_f) :- at_idx(o, 6), at_idx(b, 29), at_idx(y, 53).
onface(yob_c, b, w_f) :- at_idx(b, 6), at_idx(y, 29), at_idx(o, 53).
onface(yob_c, y, r_f) :- at_idx(b, 6), at_idx(y, 29), at_idx(o, 53).
onface(yob_c, o, g_f) :- at_idx(b, 6), at_idx(y, 29), at_idx(o, 53).
onface(yob_c, b, w_f) :- at_idx(b, 6), at_idx(o, 29), at_idx(y, 53).
onface(yob_c, o, r_f) :- at_idx(b, 6), at_idx(o, 29), at_idx(y, 53).
onface(yob_c, y, g_f) :- at_idx(b, 6), at_idx(o, 29), at_idx(y, 53).
onface(yog_c, y, w_f) :- at_idx(y, 6), at_idx(o, 29), at_idx(g, 53).
onface(yog_c, o, r_f) :- at_idx(y, 6), at_idx(o, 29), at_idx(g, 53).
onface(yog_c, g, g_f) :- at_idx(y, 6), at_idx(o, 29), at_idx(g, 53).
onface(yog_c, y, w_f) :- at_idx(y, 6), at_idx(g, 29), at_idx(o, 53).
onface(yog_c, g, r_f) :- at_idx(y, 6), at_idx(g, 29), at_idx(o, 53).
onface(yog_c, o, g_f) :- at_idx(y, 6), at_idx(g, 29), at_idx(o, 53).
onface(yog_c, o, w_f) :- at_idx(o, 6), at_idx(y, 29), at_idx(g, 53).
onface(yog_c, y, r_f) :- at_idx(o, 6), at_idx(y, 29), at_idx(g, 53).
onface(yog_c, g, g_f) :- at_idx(o, 6), at_idx(y, 29), at_idx(g, 53).
onface(yog_c, o, w_f) :- at_idx(o, 6), at_idx(g, 29), at_idx(y, 53).
onface(yog_c, g, r_f) :- at_idx(o, 6), at_idx(g, 29), at_idx(y, 53).
onface(yog_c, y, g_f) :- at_idx(o, 6), at_idx(g, 29), at_idx(y, 53).
onface(yog_c, g, w_f) :- at_idx(g, 6), at_idx(y, 29), at_idx(o, 53).
onface(yog_c, y, r_f) :- at_idx(g, 6), at_idx(y, 29), at_idx(o, 53).
onface(yog_c, o, g_f) :- at_idx(g, 6), at_idx(y, 29), at_idx(o, 53).
onface(yog_c, g, w_f) :- at_idx(g, 6), at_idx(o, 29), at_idx(y, 53).
onface(yog_c, o, r_f) :- at_idx(g, 6), at_idx(o, 29), at_idx(y, 53).
onface(yog_c, y, g_f) :- at_idx(g, 6), at_idx(o, 29), at_idx(y, 53).
onface(yrb_c, y, w_f) :- at_idx(y, 6), at_idx(r, 29), at_idx(b, 53).
onface(yrb_c, r, r_f) :- at_idx(y, 6), at_idx(r, 29), at_idx(b, 53).
onface(yrb_c, b, g_f) :- at_idx(y, 6), at_idx(r, 29), at_idx(b, 53).
onface(yrb_c, y, w_f) :- at_idx(y, 6), at_idx(b, 29), at_idx(r, 53).
onface(yrb_c, b, r_f) :- at_idx(y, 6), at_idx(b, 29), at_idx(r, 53).
onface(yrb_c, r, g_f) :- at_idx(y, 6), at_idx(b, 29), at_idx(r, 53).
onface(yrb_c, r, w_f) :- at_idx(r, 6), at_idx(y, 29), at_idx(b, 53).
onface(yrb_c, y, r_f) :- at_idx(r, 6), at_idx(y, 29), at_idx(b, 53).
onface(yrb_c, b, g_f) :- at_idx(r, 6), at_idx(y, 29), at_idx(b, 53).
onface(yrb_c, r, w_f) :- at_idx(r, 6), at_idx(b, 29), at_idx(y, 53).
onface(yrb_c, b, r_f) :- at_idx(r, 6), at_idx(b, 29), at_idx(y, 53).
onface(yrb_c, y, g_f) :- at_idx(r, 6), at_idx(b, 29), at_idx(y, 53).
onface(yrb_c, b, w_f) :- at_idx(b, 6), at_idx(y, 29), at_idx(r, 53).
onface(yrb_c, y, r_f) :- at_idx(b, 6), at_idx(y, 29), at_idx(r, 53).
onface(yrb_c, r, g_f) :- at_idx(b, 6), at_idx(y, 29), at_idx(r, 53).
onface(yrb_c, b, w_f) :- at_idx(b, 6), at_idx(r, 29), at_idx(y, 53).
onface(yrb_c, r, r_f) :- at_idx(b, 6), at_idx(r, 29), at_idx(y, 53).
onface(yrb_c, y, g_f) :- at_idx(b, 6), at_idx(r, 29), at_idx(y, 53).
onface(yrg_c, y, w_f) :- at_idx(y, 6), at_idx(r, 29), at_idx(g, 53).
onface(yrg_c, r, r_f) :- at_idx(y, 6), at_idx(r, 29), at_idx(g, 53).
onface(yrg_c, g, g_f) :- at_idx(y, 6), at_idx(r, 29), at_idx(g, 53).
onface(yrg_c, y, w_f) :- at_idx(y, 6), at_idx(g, 29), at_idx(r, 53).
onface(yrg_c, g, r_f) :- at_idx(y, 6), at_idx(g, 29), at_idx(r, 53).
onface(yrg_c, r, g_f) :- at_idx(y, 6), at_idx(g, 29), at_idx(r, 53).
onface(yrg_c, r, w_f) :- at_idx(r, 6), at_idx(y, 29), at_idx(g, 53).
onface(yrg_c, y, r_f) :- at_idx(r, 6), at_idx(y, 29), at_idx(g, 53).
onface(yrg_c, g, g_f) :- at_idx(r, 6), at_idx(y, 29), at_idx(g, 53).
onface(yrg_c, r, w_f) :- at_idx(r, 6), at_idx(g, 29), at_idx(y, 53).
onface(yrg_c, g, r_f) :- at_idx(r, 6), at_idx(g, 29), at_idx(y, 53).
onface(yrg_c, y, g_f) :- at_idx(r, 6), at_idx(g, 29), at_idx(y, 53).
onface(yrg_c, g, w_f) :- at_idx(g, 6), at_idx(y, 29), at_idx(r, 53).
onface(yrg_c, y, r_f) :- at_idx(g, 6), at_idx(y, 29), at_idx(r, 53).
onface(yrg_c, r, g_f) :- at_idx(g, 6), at_idx(y, 29), at_idx(r, 53).
onface(yrg_c, g, w_f) :- at_idx(g, 6), at_idx(r, 29), at_idx(y, 53).
onface(yrg_c, r, r_f) :- at_idx(g, 6), at_idx(r, 29), at_idx(y, 53).
onface(yrg_c, y, g_f) :- at_idx(g, 6), at_idx(r, 29), at_idx(y, 53).
onface(wog_c, w, w_f) :- at_idx(w, 8), at_idx(o, 35), at_idx(g, 38).
onface(wog_c, o, r_f) :- at_idx(w, 8), at_idx(o, 35), at_idx(g, 38).
onface(wog_c, g, b_f) :- at_idx(w, 8), at_idx(o, 35), at_idx(g, 38).
onface(wog_c, w, w_f) :- at_idx(w, 8), at_idx(g, 35), at_idx(o, 38).
onface(wog_c, g, r_f) :- at_idx(w, 8), at_idx(g, 35), at_idx(o, 38).
onface(wog_c, o, b_f) :- at_idx(w, 8), at_idx(g, 35), at_idx(o, 38).
onface(wog_c, o, w_f) :- at_idx(o, 8), at_idx(w, 35), at_idx(g, 38).
onface(wog_c, w, r_f) :- at_idx(o, 8), at_idx(w, 35), at_idx(g, 38).
onface(wog_c, g, b_f) :- at_idx(o, 8), at_idx(w, 35), at_idx(g, 38).
onface(wog_c, o, w_f) :- at_idx(o, 8), at_idx(g, 35), at_idx(w, 38).
onface(wog_c, g, r_f) :- at_idx(o, 8), at_idx(g, 35), at_idx(w, 38).
onface(wog_c, w, b_f) :- at_idx(o, 8), at_idx(g, 35), at_idx(w, 38).
onface(wog_c, g, w_f) :- at_idx(g, 8), at_idx(w, 35), at_idx(o, 38).
onface(wog_c, w, r_f) :- at_idx(g, 8), at_idx(w, 35), at_idx(o, 38).
onface(wog_c, o, b_f) :- at_idx(g, 8), at_idx(w, 35), at_idx(o, 38).
onface(wog_c, g, w_f) :- at_idx(g, 8), at_idx(o, 35), at_idx(w, 38).
onface(wog_c, o, r_f) :- at_idx(g, 8), at_idx(o, 35), at_idx(w, 38).
onface(wog_c, w, b_f) :- at_idx(g, 8), at_idx(o, 35), at_idx(w, 38).
onface(wob_c, w, w_f) :- at_idx(w, 8), at_idx(o, 35), at_idx(b, 38).
onface(wob_c, o, r_f) :- at_idx(w, 8), at_idx(o, 35), at_idx(b, 38).
onface(wob_c, b, b_f) :- at_idx(w, 8), at_idx(o, 35), at_idx(b, 38).
onface(wob_c, w, w_f) :- at_idx(w, 8), at_idx(b, 35), at_idx(o, 38).
onface(wob_c, b, r_f) :- at_idx(w, 8), at_idx(b, 35), at_idx(o, 38).
onface(wob_c, o, b_f) :- at_idx(w, 8), at_idx(b, 35), at_idx(o, 38).
onface(wob_c, o, w_f) :- at_idx(o, 8), at_idx(w, 35), at_idx(b, 38).
onface(wob_c, w, r_f) :- at_idx(o, 8), at_idx(w, 35), at_idx(b, 38).
onface(wob_c, b, b_f) :- at_idx(o, 8), at_idx(w, 35), at_idx(b, 38).
onface(wob_c, o, w_f) :- at_idx(o, 8), at_idx(b, 35), at_idx(w, 38).
onface(wob_c, b, r_f) :- at_idx(o, 8), at_idx(b, 35), at_idx(w, 38).
onface(wob_c, w, b_f) :- at_idx(o, 8), at_idx(b, 35), at_idx(w, 38).
onface(wob_c, b, w_f) :- at_idx(b, 8), at_idx(w, 35), at_idx(o, 38).
onface(wob_c, w, r_f) :- at_idx(b, 8), at_idx(w, 35), at_idx(o, 38).
onface(wob_c, o, b_f) :- at_idx(b, 8), at_idx(w, 35), at_idx(o, 38).
onface(wob_c, b, w_f) :- at_idx(b, 8), at_idx(o, 35), at_idx(w, 38).
onface(wob_c, o, r_f) :- at_idx(b, 8), at_idx(o, 35), at_idx(w, 38).
onface(wob_c, w, b_f) :- at_idx(b, 8), at_idx(o, 35), at_idx(w, 38).
onface(wrg_c, w, w_f) :- at_idx(w, 8), at_idx(r, 35), at_idx(g, 38).
onface(wrg_c, r, r_f) :- at_idx(w, 8), at_idx(r, 35), at_idx(g, 38).
onface(wrg_c, g, b_f) :- at_idx(w, 8), at_idx(r, 35), at_idx(g, 38).
onface(wrg_c, w, w_f) :- at_idx(w, 8), at_idx(g, 35), at_idx(r, 38).
onface(wrg_c, g, r_f) :- at_idx(w, 8), at_idx(g, 35), at_idx(r, 38).
onface(wrg_c, r, b_f) :- at_idx(w, 8), at_idx(g, 35), at_idx(r, 38).
onface(wrg_c, r, w_f) :- at_idx(r, 8), at_idx(w, 35), at_idx(g, 38).
onface(wrg_c, w, r_f) :- at_idx(r, 8), at_idx(w, 35), at_idx(g, 38).
onface(wrg_c, g, b_f) :- at_idx(r, 8), at_idx(w, 35), at_idx(g, 38).
onface(wrg_c, r, w_f) :- at_idx(r, 8), at_idx(g, 35), at_idx(w, 38).
onface(wrg_c, g, r_f) :- at_idx(r, 8), at_idx(g, 35), at_idx(w, 38).
onface(wrg_c, w, b_f) :- at_idx(r, 8), at_idx(g, 35), at_idx(w, 38).
onface(wrg_c, g, w_f) :- at_idx(g, 8), at_idx(w, 35), at_idx(r, 38).
onface(wrg_c, w, r_f) :- at_idx(g, 8), at_idx(w, 35), at_idx(r, 38).
onface(wrg_c, r, b_f) :- at_idx(g, 8), at_idx(w, 35), at_idx(r, 38).
onface(wrg_c, g, w_f) :- at_idx(g, 8), at_idx(r, 35), at_idx(w, 38).
onface(wrg_c, r, r_f) :- at_idx(g, 8), at_idx(r, 35), at_idx(w, 38).
onface(wrg_c, w, b_f) :- at_idx(g, 8), at_idx(r, 35), at_idx(w, 38).
onface(wrb_c, w, w_f) :- at_idx(w, 8), at_idx(r, 35), at_idx(b, 38).
onface(wrb_c, r, r_f) :- at_idx(w, 8), at_idx(r, 35), at_idx(b, 38).
onface(wrb_c, b, b_f) :- at_idx(w, 8), at_idx(r, 35), at_idx(b, 38).
onface(wrb_c, w, w_f) :- at_idx(w, 8), at_idx(b, 35), at_idx(r, 38).
onface(wrb_c, b, r_f) :- at_idx(w, 8), at_idx(b, 35), at_idx(r, 38).
onface(wrb_c, r, b_f) :- at_idx(w, 8), at_idx(b, 35), at_idx(r, 38).
onface(wrb_c, r, w_f) :- at_idx(r, 8), at_idx(w, 35), at_idx(b, 38).
onface(wrb_c, w, r_f) :- at_idx(r, 8), at_idx(w, 35), at_idx(b, 38).
onface(wrb_c, b, b_f) :- at_idx(r, 8), at_idx(w, 35), at_idx(b, 38).
onface(wrb_c, r, w_f) :- at_idx(r, 8), at_idx(b, 35), at_idx(w, 38).
onface(wrb_c, b, r_f) :- at_idx(r, 8), at_idx(b, 35), at_idx(w, 38).
onface(wrb_c, w, b_f) :- at_idx(r, 8), at_idx(b, 35), at_idx(w, 38).
onface(wrb_c, b, w_f) :- at_idx(b, 8), at_idx(w, 35), at_idx(r, 38).
onface(wrb_c, w, r_f) :- at_idx(b, 8), at_idx(w, 35), at_idx(r, 38).
onface(wrb_c, r, b_f) :- at_idx(b, 8), at_idx(w, 35), at_idx(r, 38).
onface(wrb_c, b, w_f) :- at_idx(b, 8), at_idx(r, 35), at_idx(w, 38).
onface(wrb_c, r, r_f) :- at_idx(b, 8), at_idx(r, 35), at_idx(w, 38).
onface(wrb_c, w, b_f) :- at_idx(b, 8), at_idx(r, 35), at_idx(w, 38).
onface(yob_c, y, w_f) :- at_idx(y, 8), at_idx(o, 35), at_idx(b, 38).
onface(yob_c, o, r_f) :- at_idx(y, 8), at_idx(o, 35), at_idx(b, 38).
onface(yob_c, b, b_f) :- at_idx(y, 8), at_idx(o, 35), at_idx(b, 38).
onface(yob_c, y, w_f) :- at_idx(y, 8), at_idx(b, 35), at_idx(o, 38).
onface(yob_c, b, r_f) :- at_idx(y, 8), at_idx(b, 35), at_idx(o, 38).
onface(yob_c, o, b_f) :- at_idx(y, 8), at_idx(b, 35), at_idx(o, 38).
onface(yob_c, o, w_f) :- at_idx(o, 8), at_idx(y, 35), at_idx(b, 38).
onface(yob_c, y, r_f) :- at_idx(o, 8), at_idx(y, 35), at_idx(b, 38).
onface(yob_c, b, b_f) :- at_idx(o, 8), at_idx(y, 35), at_idx(b, 38).
onface(yob_c, o, w_f) :- at_idx(o, 8), at_idx(b, 35), at_idx(y, 38).
onface(yob_c, b, r_f) :- at_idx(o, 8), at_idx(b, 35), at_idx(y, 38).
onface(yob_c, y, b_f) :- at_idx(o, 8), at_idx(b, 35), at_idx(y, 38).
onface(yob_c, b, w_f) :- at_idx(b, 8), at_idx(y, 35), at_idx(o, 38).
onface(yob_c, y, r_f) :- at_idx(b, 8), at_idx(y, 35), at_idx(o, 38).
onface(yob_c, o, b_f) :- at_idx(b, 8), at_idx(y, 35), at_idx(o, 38).
onface(yob_c, b, w_f) :- at_idx(b, 8), at_idx(o, 35), at_idx(y, 38).
onface(yob_c, o, r_f) :- at_idx(b, 8), at_idx(o, 35), at_idx(y, 38).
onface(yob_c, y, b_f) :- at_idx(b, 8), at_idx(o, 35), at_idx(y, 38).
onface(yog_c, y, w_f) :- at_idx(y, 8), at_idx(o, 35), at_idx(g, 38).
onface(yog_c, o, r_f) :- at_idx(y, 8), at_idx(o, 35), at_idx(g, 38).
onface(yog_c, g, b_f) :- at_idx(y, 8), at_idx(o, 35), at_idx(g, 38).
onface(yog_c, y, w_f) :- at_idx(y, 8), at_idx(g, 35), at_idx(o, 38).
onface(yog_c, g, r_f) :- at_idx(y, 8), at_idx(g, 35), at_idx(o, 38).
onface(yog_c, o, b_f) :- at_idx(y, 8), at_idx(g, 35), at_idx(o, 38).
onface(yog_c, o, w_f) :- at_idx(o, 8), at_idx(y, 35), at_idx(g, 38).
onface(yog_c, y, r_f) :- at_idx(o, 8), at_idx(y, 35), at_idx(g, 38).
onface(yog_c, g, b_f) :- at_idx(o, 8), at_idx(y, 35), at_idx(g, 38).
onface(yog_c, o, w_f) :- at_idx(o, 8), at_idx(g, 35), at_idx(y, 38).
onface(yog_c, g, r_f) :- at_idx(o, 8), at_idx(g, 35), at_idx(y, 38).
onface(yog_c, y, b_f) :- at_idx(o, 8), at_idx(g, 35), at_idx(y, 38).
onface(yog_c, g, w_f) :- at_idx(g, 8), at_idx(y, 35), at_idx(o, 38).
onface(yog_c, y, r_f) :- at_idx(g, 8), at_idx(y, 35), at_idx(o, 38).
onface(yog_c, o, b_f) :- at_idx(g, 8), at_idx(y, 35), at_idx(o, 38).
onface(yog_c, g, w_f) :- at_idx(g, 8), at_idx(o, 35), at_idx(y, 38).
onface(yog_c, o, r_f) :- at_idx(g, 8), at_idx(o, 35), at_idx(y, 38).
onface(yog_c, y, b_f) :- at_idx(g, 8), at_idx(o, 35), at_idx(y, 38).
onface(yrb_c, y, w_f) :- at_idx(y, 8), at_idx(r, 35), at_idx(b, 38).
onface(yrb_c, r, r_f) :- at_idx(y, 8), at_idx(r, 35), at_idx(b, 38).
onface(yrb_c, b, b_f) :- at_idx(y, 8), at_idx(r, 35), at_idx(b, 38).
onface(yrb_c, y, w_f) :- at_idx(y, 8), at_idx(b, 35), at_idx(r, 38).
onface(yrb_c, b, r_f) :- at_idx(y, 8), at_idx(b, 35), at_idx(r, 38).
onface(yrb_c, r, b_f) :- at_idx(y, 8), at_idx(b, 35), at_idx(r, 38).
onface(yrb_c, r, w_f) :- at_idx(r, 8), at_idx(y, 35), at_idx(b, 38).
onface(yrb_c, y, r_f) :- at_idx(r, 8), at_idx(y, 35), at_idx(b, 38).
onface(yrb_c, b, b_f) :- at_idx(r, 8), at_idx(y, 35), at_idx(b, 38).
onface(yrb_c, r, w_f) :- at_idx(r, 8), at_idx(b, 35), at_idx(y, 38).
onface(yrb_c, b, r_f) :- at_idx(r, 8), at_idx(b, 35), at_idx(y, 38).
onface(yrb_c, y, b_f) :- at_idx(r, 8), at_idx(b, 35), at_idx(y, 38).
onface(yrb_c, b, w_f) :- at_idx(b, 8), at_idx(y, 35), at_idx(r, 38).
onface(yrb_c, y, r_f) :- at_idx(b, 8), at_idx(y, 35), at_idx(r, 38).
onface(yrb_c, r, b_f) :- at_idx(b, 8), at_idx(y, 35), at_idx(r, 38).
onface(yrb_c, b, w_f) :- at_idx(b, 8), at_idx(r, 35), at_idx(y, 38).
onface(yrb_c, r, r_f) :- at_idx(b, 8), at_idx(r, 35), at_idx(y, 38).
onface(yrb_c, y, b_f) :- at_idx(b, 8), at_idx(r, 35), at_idx(y, 38).
onface(yrg_c, y, w_f) :- at_idx(y, 8), at_idx(r, 35), at_idx(g, 38).
onface(yrg_c, r, r_f) :- at_idx(y, 8), at_idx(r, 35), at_idx(g, 38).
onface(yrg_c, g, b_f) :- at_idx(y, 8), at_idx(r, 35), at_idx(g, 38).
onface(yrg_c, y, w_f) :- at_idx(y, 8), at_idx(g, 35), at_idx(r, 38).
onface(yrg_c, g, r_f) :- at_idx(y, 8), at_idx(g, 35), at_idx(r, 38).
onface(yrg_c, r, b_f) :- at_idx(y, 8), at_idx(g, 35), at_idx(r, 38).
onface(yrg_c, r, w_f) :- at_idx(r, 8), at_idx(y, 35), at_idx(g, 38).
onface(yrg_c, y, r_f) :- at_idx(r, 8), at_idx(y, 35), at_idx(g, 38).
onface(yrg_c, g, b_f) :- at_idx(r, 8), at_idx(y, 35), at_idx(g, 38).
onface(yrg_c, r, w_f) :- at_idx(r, 8), at_idx(g, 35), at_idx(y, 38).
onface(yrg_c, g, r_f) :- at_idx(r, 8), at_idx(g, 35), at_idx(y, 38).
onface(yrg_c, y, b_f) :- at_idx(r, 8), at_idx(g, 35), at_idx(y, 38).
onface(yrg_c, g, w_f) :- at_idx(g, 8), at_idx(y, 35), at_idx(r, 38).
onface(yrg_c, y, r_f) :- at_idx(g, 8), at_idx(y, 35), at_idx(r, 38).
onface(yrg_c, r, b_f) :- at_idx(g, 8), at_idx(y, 35), at_idx(r, 38).
onface(yrg_c, g, w_f) :- at_idx(g, 8), at_idx(r, 35), at_idx(y, 38).
onface(yrg_c, r, r_f) :- at_idx(g, 8), at_idx(r, 35), at_idx(y, 38).
onface(yrg_c, y, b_f) :- at_idx(g, 8), at_idx(r, 35), at_idx(y, 38).
onface(wog_c, w, y_f) :- at_idx(w, 9), at_idx(o, 18), at_idx(g, 42).
onface(wog_c, o, o_f) :- at_idx(w, 9), at_idx(o, 18), at_idx(g, 42).
onface(wog_c, g, b_f) :- at_idx(w, 9), at_idx(o, 18), at_idx(g, 42).
onface(wog_c, w, y_f) :- at_idx(w, 9), at_idx(g, 18), at_idx(o, 42).
onface(wog_c, g, o_f) :- at_idx(w, 9), at_idx(g, 18), at_idx(o, 42).
onface(wog_c, o, b_f) :- at_idx(w, 9), at_idx(g, 18), at_idx(o, 42).
onface(wog_c, o, y_f) :- at_idx(o, 9), at_idx(w, 18), at_idx(g, 42).
onface(wog_c, w, o_f) :- at_idx(o, 9), at_idx(w, 18), at_idx(g, 42).
onface(wog_c, g, b_f) :- at_idx(o, 9), at_idx(w, 18), at_idx(g, 42).
onface(wog_c, o, y_f) :- at_idx(o, 9), at_idx(g, 18), at_idx(w, 42).
onface(wog_c, g, o_f) :- at_idx(o, 9), at_idx(g, 18), at_idx(w, 42).
onface(wog_c, w, b_f) :- at_idx(o, 9), at_idx(g, 18), at_idx(w, 42).
onface(wog_c, g, y_f) :- at_idx(g, 9), at_idx(w, 18), at_idx(o, 42).
onface(wog_c, w, o_f) :- at_idx(g, 9), at_idx(w, 18), at_idx(o, 42).
onface(wog_c, o, b_f) :- at_idx(g, 9), at_idx(w, 18), at_idx(o, 42).
onface(wog_c, g, y_f) :- at_idx(g, 9), at_idx(o, 18), at_idx(w, 42).
onface(wog_c, o, o_f) :- at_idx(g, 9), at_idx(o, 18), at_idx(w, 42).
onface(wog_c, w, b_f) :- at_idx(g, 9), at_idx(o, 18), at_idx(w, 42).
onface(wob_c, w, y_f) :- at_idx(w, 9), at_idx(o, 18), at_idx(b, 42).
onface(wob_c, o, o_f) :- at_idx(w, 9), at_idx(o, 18), at_idx(b, 42).
onface(wob_c, b, b_f) :- at_idx(w, 9), at_idx(o, 18), at_idx(b, 42).
onface(wob_c, w, y_f) :- at_idx(w, 9), at_idx(b, 18), at_idx(o, 42).
onface(wob_c, b, o_f) :- at_idx(w, 9), at_idx(b, 18), at_idx(o, 42).
onface(wob_c, o, b_f) :- at_idx(w, 9), at_idx(b, 18), at_idx(o, 42).
onface(wob_c, o, y_f) :- at_idx(o, 9), at_idx(w, 18), at_idx(b, 42).
onface(wob_c, w, o_f) :- at_idx(o, 9), at_idx(w, 18), at_idx(b, 42).
onface(wob_c, b, b_f) :- at_idx(o, 9), at_idx(w, 18), at_idx(b, 42).
onface(wob_c, o, y_f) :- at_idx(o, 9), at_idx(b, 18), at_idx(w, 42).
onface(wob_c, b, o_f) :- at_idx(o, 9), at_idx(b, 18), at_idx(w, 42).
onface(wob_c, w, b_f) :- at_idx(o, 9), at_idx(b, 18), at_idx(w, 42).
onface(wob_c, b, y_f) :- at_idx(b, 9), at_idx(w, 18), at_idx(o, 42).
onface(wob_c, w, o_f) :- at_idx(b, 9), at_idx(w, 18), at_idx(o, 42).
onface(wob_c, o, b_f) :- at_idx(b, 9), at_idx(w, 18), at_idx(o, 42).
onface(wob_c, b, y_f) :- at_idx(b, 9), at_idx(o, 18), at_idx(w, 42).
onface(wob_c, o, o_f) :- at_idx(b, 9), at_idx(o, 18), at_idx(w, 42).
onface(wob_c, w, b_f) :- at_idx(b, 9), at_idx(o, 18), at_idx(w, 42).
onface(wrg_c, w, y_f) :- at_idx(w, 9), at_idx(r, 18), at_idx(g, 42).
onface(wrg_c, r, o_f) :- at_idx(w, 9), at_idx(r, 18), at_idx(g, 42).
onface(wrg_c, g, b_f) :- at_idx(w, 9), at_idx(r, 18), at_idx(g, 42).
onface(wrg_c, w, y_f) :- at_idx(w, 9), at_idx(g, 18), at_idx(r, 42).
onface(wrg_c, g, o_f) :- at_idx(w, 9), at_idx(g, 18), at_idx(r, 42).
onface(wrg_c, r, b_f) :- at_idx(w, 9), at_idx(g, 18), at_idx(r, 42).
onface(wrg_c, r, y_f) :- at_idx(r, 9), at_idx(w, 18), at_idx(g, 42).
onface(wrg_c, w, o_f) :- at_idx(r, 9), at_idx(w, 18), at_idx(g, 42).
onface(wrg_c, g, b_f) :- at_idx(r, 9), at_idx(w, 18), at_idx(g, 42).
onface(wrg_c, r, y_f) :- at_idx(r, 9), at_idx(g, 18), at_idx(w, 42).
onface(wrg_c, g, o_f) :- at_idx(r, 9), at_idx(g, 18), at_idx(w, 42).
onface(wrg_c, w, b_f) :- at_idx(r, 9), at_idx(g, 18), at_idx(w, 42).
onface(wrg_c, g, y_f) :- at_idx(g, 9), at_idx(w, 18), at_idx(r, 42).
onface(wrg_c, w, o_f) :- at_idx(g, 9), at_idx(w, 18), at_idx(r, 42).
onface(wrg_c, r, b_f) :- at_idx(g, 9), at_idx(w, 18), at_idx(r, 42).
onface(wrg_c, g, y_f) :- at_idx(g, 9), at_idx(r, 18), at_idx(w, 42).
onface(wrg_c, r, o_f) :- at_idx(g, 9), at_idx(r, 18), at_idx(w, 42).
onface(wrg_c, w, b_f) :- at_idx(g, 9), at_idx(r, 18), at_idx(w, 42).
onface(wrb_c, w, y_f) :- at_idx(w, 9), at_idx(r, 18), at_idx(b, 42).
onface(wrb_c, r, o_f) :- at_idx(w, 9), at_idx(r, 18), at_idx(b, 42).
onface(wrb_c, b, b_f) :- at_idx(w, 9), at_idx(r, 18), at_idx(b, 42).
onface(wrb_c, w, y_f) :- at_idx(w, 9), at_idx(b, 18), at_idx(r, 42).
onface(wrb_c, b, o_f) :- at_idx(w, 9), at_idx(b, 18), at_idx(r, 42).
onface(wrb_c, r, b_f) :- at_idx(w, 9), at_idx(b, 18), at_idx(r, 42).
onface(wrb_c, r, y_f) :- at_idx(r, 9), at_idx(w, 18), at_idx(b, 42).
onface(wrb_c, w, o_f) :- at_idx(r, 9), at_idx(w, 18), at_idx(b, 42).
onface(wrb_c, b, b_f) :- at_idx(r, 9), at_idx(w, 18), at_idx(b, 42).
onface(wrb_c, r, y_f) :- at_idx(r, 9), at_idx(b, 18), at_idx(w, 42).
onface(wrb_c, b, o_f) :- at_idx(r, 9), at_idx(b, 18), at_idx(w, 42).
onface(wrb_c, w, b_f) :- at_idx(r, 9), at_idx(b, 18), at_idx(w, 42).
onface(wrb_c, b, y_f) :- at_idx(b, 9), at_idx(w, 18), at_idx(r, 42).
onface(wrb_c, w, o_f) :- at_idx(b, 9), at_idx(w, 18), at_idx(r, 42).
onface(wrb_c, r, b_f) :- at_idx(b, 9), at_idx(w, 18), at_idx(r, 42).
onface(wrb_c, b, y_f) :- at_idx(b, 9), at_idx(r, 18), at_idx(w, 42).
onface(wrb_c, r, o_f) :- at_idx(b, 9), at_idx(r, 18), at_idx(w, 42).
onface(wrb_c, w, b_f) :- at_idx(b, 9), at_idx(r, 18), at_idx(w, 42).
onface(yob_c, y, y_f) :- at_idx(y, 9), at_idx(o, 18), at_idx(b, 42).
onface(yob_c, o, o_f) :- at_idx(y, 9), at_idx(o, 18), at_idx(b, 42).
onface(yob_c, b, b_f) :- at_idx(y, 9), at_idx(o, 18), at_idx(b, 42).
onface(yob_c, y, y_f) :- at_idx(y, 9), at_idx(b, 18), at_idx(o, 42).
onface(yob_c, b, o_f) :- at_idx(y, 9), at_idx(b, 18), at_idx(o, 42).
onface(yob_c, o, b_f) :- at_idx(y, 9), at_idx(b, 18), at_idx(o, 42).
onface(yob_c, o, y_f) :- at_idx(o, 9), at_idx(y, 18), at_idx(b, 42).
onface(yob_c, y, o_f) :- at_idx(o, 9), at_idx(y, 18), at_idx(b, 42).
onface(yob_c, b, b_f) :- at_idx(o, 9), at_idx(y, 18), at_idx(b, 42).
onface(yob_c, o, y_f) :- at_idx(o, 9), at_idx(b, 18), at_idx(y, 42).
onface(yob_c, b, o_f) :- at_idx(o, 9), at_idx(b, 18), at_idx(y, 42).
onface(yob_c, y, b_f) :- at_idx(o, 9), at_idx(b, 18), at_idx(y, 42).
onface(yob_c, b, y_f) :- at_idx(b, 9), at_idx(y, 18), at_idx(o, 42).
onface(yob_c, y, o_f) :- at_idx(b, 9), at_idx(y, 18), at_idx(o, 42).
onface(yob_c, o, b_f) :- at_idx(b, 9), at_idx(y, 18), at_idx(o, 42).
onface(yob_c, b, y_f) :- at_idx(b, 9), at_idx(o, 18), at_idx(y, 42).
onface(yob_c, o, o_f) :- at_idx(b, 9), at_idx(o, 18), at_idx(y, 42).
onface(yob_c, y, b_f) :- at_idx(b, 9), at_idx(o, 18), at_idx(y, 42).
onface(yog_c, y, y_f) :- at_idx(y, 9), at_idx(o, 18), at_idx(g, 42).
onface(yog_c, o, o_f) :- at_idx(y, 9), at_idx(o, 18), at_idx(g, 42).
onface(yog_c, g, b_f) :- at_idx(y, 9), at_idx(o, 18), at_idx(g, 42).
onface(yog_c, y, y_f) :- at_idx(y, 9), at_idx(g, 18), at_idx(o, 42).
onface(yog_c, g, o_f) :- at_idx(y, 9), at_idx(g, 18), at_idx(o, 42).
onface(yog_c, o, b_f) :- at_idx(y, 9), at_idx(g, 18), at_idx(o, 42).
onface(yog_c, o, y_f) :- at_idx(o, 9), at_idx(y, 18), at_idx(g, 42).
onface(yog_c, y, o_f) :- at_idx(o, 9), at_idx(y, 18), at_idx(g, 42).
onface(yog_c, g, b_f) :- at_idx(o, 9), at_idx(y, 18), at_idx(g, 42).
onface(yog_c, o, y_f) :- at_idx(o, 9), at_idx(g, 18), at_idx(y, 42).
onface(yog_c, g, o_f) :- at_idx(o, 9), at_idx(g, 18), at_idx(y, 42).
onface(yog_c, y, b_f) :- at_idx(o, 9), at_idx(g, 18), at_idx(y, 42).
onface(yog_c, g, y_f) :- at_idx(g, 9), at_idx(y, 18), at_idx(o, 42).
onface(yog_c, y, o_f) :- at_idx(g, 9), at_idx(y, 18), at_idx(o, 42).
onface(yog_c, o, b_f) :- at_idx(g, 9), at_idx(y, 18), at_idx(o, 42).
onface(yog_c, g, y_f) :- at_idx(g, 9), at_idx(o, 18), at_idx(y, 42).
onface(yog_c, o, o_f) :- at_idx(g, 9), at_idx(o, 18), at_idx(y, 42).
onface(yog_c, y, b_f) :- at_idx(g, 9), at_idx(o, 18), at_idx(y, 42).
onface(yrb_c, y, y_f) :- at_idx(y, 9), at_idx(r, 18), at_idx(b, 42).
onface(yrb_c, r, o_f) :- at_idx(y, 9), at_idx(r, 18), at_idx(b, 42).
onface(yrb_c, b, b_f) :- at_idx(y, 9), at_idx(r, 18), at_idx(b, 42).
onface(yrb_c, y, y_f) :- at_idx(y, 9), at_idx(b, 18), at_idx(r, 42).
onface(yrb_c, b, o_f) :- at_idx(y, 9), at_idx(b, 18), at_idx(r, 42).
onface(yrb_c, r, b_f) :- at_idx(y, 9), at_idx(b, 18), at_idx(r, 42).
onface(yrb_c, r, y_f) :- at_idx(r, 9), at_idx(y, 18), at_idx(b, 42).
onface(yrb_c, y, o_f) :- at_idx(r, 9), at_idx(y, 18), at_idx(b, 42).
onface(yrb_c, b, b_f) :- at_idx(r, 9), at_idx(y, 18), at_idx(b, 42).
onface(yrb_c, r, y_f) :- at_idx(r, 9), at_idx(b, 18), at_idx(y, 42).
onface(yrb_c, b, o_f) :- at_idx(r, 9), at_idx(b, 18), at_idx(y, 42).
onface(yrb_c, y, b_f) :- at_idx(r, 9), at_idx(b, 18), at_idx(y, 42).
onface(yrb_c, b, y_f) :- at_idx(b, 9), at_idx(y, 18), at_idx(r, 42).
onface(yrb_c, y, o_f) :- at_idx(b, 9), at_idx(y, 18), at_idx(r, 42).
onface(yrb_c, r, b_f) :- at_idx(b, 9), at_idx(y, 18), at_idx(r, 42).
onface(yrb_c, b, y_f) :- at_idx(b, 9), at_idx(r, 18), at_idx(y, 42).
onface(yrb_c, r, o_f) :- at_idx(b, 9), at_idx(r, 18), at_idx(y, 42).
onface(yrb_c, y, b_f) :- at_idx(b, 9), at_idx(r, 18), at_idx(y, 42).
onface(yrg_c, y, y_f) :- at_idx(y, 9), at_idx(r, 18), at_idx(g, 42).
onface(yrg_c, r, o_f) :- at_idx(y, 9), at_idx(r, 18), at_idx(g, 42).
onface(yrg_c, g, b_f) :- at_idx(y, 9), at_idx(r, 18), at_idx(g, 42).
onface(yrg_c, y, y_f) :- at_idx(y, 9), at_idx(g, 18), at_idx(r, 42).
onface(yrg_c, g, o_f) :- at_idx(y, 9), at_idx(g, 18), at_idx(r, 42).
onface(yrg_c, r, b_f) :- at_idx(y, 9), at_idx(g, 18), at_idx(r, 42).
onface(yrg_c, r, y_f) :- at_idx(r, 9), at_idx(y, 18), at_idx(g, 42).
onface(yrg_c, y, o_f) :- at_idx(r, 9), at_idx(y, 18), at_idx(g, 42).
onface(yrg_c, g, b_f) :- at_idx(r, 9), at_idx(y, 18), at_idx(g, 42).
onface(yrg_c, r, y_f) :- at_idx(r, 9), at_idx(g, 18), at_idx(y, 42).
onface(yrg_c, g, o_f) :- at_idx(r, 9), at_idx(g, 18), at_idx(y, 42).
onface(yrg_c, y, b_f) :- at_idx(r, 9), at_idx(g, 18), at_idx(y, 42).
onface(yrg_c, g, y_f) :- at_idx(g, 9), at_idx(y, 18), at_idx(r, 42).
onface(yrg_c, y, o_f) :- at_idx(g, 9), at_idx(y, 18), at_idx(r, 42).
onface(yrg_c, r, b_f) :- at_idx(g, 9), at_idx(y, 18), at_idx(r, 42).
onface(yrg_c, g, y_f) :- at_idx(g, 9), at_idx(r, 18), at_idx(y, 42).
onface(yrg_c, r, o_f) :- at_idx(g, 9), at_idx(r, 18), at_idx(y, 42).
onface(yrg_c, y, b_f) :- at_idx(g, 9), at_idx(r, 18), at_idx(y, 42).
onface(wog_c, w, y_f) :- at_idx(w, 11), at_idx(o, 24), at_idx(g, 45).
onface(wog_c, o, o_f) :- at_idx(w, 11), at_idx(o, 24), at_idx(g, 45).
onface(wog_c, g, g_f) :- at_idx(w, 11), at_idx(o, 24), at_idx(g, 45).
onface(wog_c, w, y_f) :- at_idx(w, 11), at_idx(g, 24), at_idx(o, 45).
onface(wog_c, g, o_f) :- at_idx(w, 11), at_idx(g, 24), at_idx(o, 45).
onface(wog_c, o, g_f) :- at_idx(w, 11), at_idx(g, 24), at_idx(o, 45).
onface(wog_c, o, y_f) :- at_idx(o, 11), at_idx(w, 24), at_idx(g, 45).
onface(wog_c, w, o_f) :- at_idx(o, 11), at_idx(w, 24), at_idx(g, 45).
onface(wog_c, g, g_f) :- at_idx(o, 11), at_idx(w, 24), at_idx(g, 45).
onface(wog_c, o, y_f) :- at_idx(o, 11), at_idx(g, 24), at_idx(w, 45).
onface(wog_c, g, o_f) :- at_idx(o, 11), at_idx(g, 24), at_idx(w, 45).
onface(wog_c, w, g_f) :- at_idx(o, 11), at_idx(g, 24), at_idx(w, 45).
onface(wog_c, g, y_f) :- at_idx(g, 11), at_idx(w, 24), at_idx(o, 45).
onface(wog_c, w, o_f) :- at_idx(g, 11), at_idx(w, 24), at_idx(o, 45).
onface(wog_c, o, g_f) :- at_idx(g, 11), at_idx(w, 24), at_idx(o, 45).
onface(wog_c, g, y_f) :- at_idx(g, 11), at_idx(o, 24), at_idx(w, 45).
onface(wog_c, o, o_f) :- at_idx(g, 11), at_idx(o, 24), at_idx(w, 45).
onface(wog_c, w, g_f) :- at_idx(g, 11), at_idx(o, 24), at_idx(w, 45).
onface(wob_c, w, y_f) :- at_idx(w, 11), at_idx(o, 24), at_idx(b, 45).
onface(wob_c, o, o_f) :- at_idx(w, 11), at_idx(o, 24), at_idx(b, 45).
onface(wob_c, b, g_f) :- at_idx(w, 11), at_idx(o, 24), at_idx(b, 45).
onface(wob_c, w, y_f) :- at_idx(w, 11), at_idx(b, 24), at_idx(o, 45).
onface(wob_c, b, o_f) :- at_idx(w, 11), at_idx(b, 24), at_idx(o, 45).
onface(wob_c, o, g_f) :- at_idx(w, 11), at_idx(b, 24), at_idx(o, 45).
onface(wob_c, o, y_f) :- at_idx(o, 11), at_idx(w, 24), at_idx(b, 45).
onface(wob_c, w, o_f) :- at_idx(o, 11), at_idx(w, 24), at_idx(b, 45).
onface(wob_c, b, g_f) :- at_idx(o, 11), at_idx(w, 24), at_idx(b, 45).
onface(wob_c, o, y_f) :- at_idx(o, 11), at_idx(b, 24), at_idx(w, 45).
onface(wob_c, b, o_f) :- at_idx(o, 11), at_idx(b, 24), at_idx(w, 45).
onface(wob_c, w, g_f) :- at_idx(o, 11), at_idx(b, 24), at_idx(w, 45).
onface(wob_c, b, y_f) :- at_idx(b, 11), at_idx(w, 24), at_idx(o, 45).
onface(wob_c, w, o_f) :- at_idx(b, 11), at_idx(w, 24), at_idx(o, 45).
onface(wob_c, o, g_f) :- at_idx(b, 11), at_idx(w, 24), at_idx(o, 45).
onface(wob_c, b, y_f) :- at_idx(b, 11), at_idx(o, 24), at_idx(w, 45).
onface(wob_c, o, o_f) :- at_idx(b, 11), at_idx(o, 24), at_idx(w, 45).
onface(wob_c, w, g_f) :- at_idx(b, 11), at_idx(o, 24), at_idx(w, 45).
onface(wrg_c, w, y_f) :- at_idx(w, 11), at_idx(r, 24), at_idx(g, 45).
onface(wrg_c, r, o_f) :- at_idx(w, 11), at_idx(r, 24), at_idx(g, 45).
onface(wrg_c, g, g_f) :- at_idx(w, 11), at_idx(r, 24), at_idx(g, 45).
onface(wrg_c, w, y_f) :- at_idx(w, 11), at_idx(g, 24), at_idx(r, 45).
onface(wrg_c, g, o_f) :- at_idx(w, 11), at_idx(g, 24), at_idx(r, 45).
onface(wrg_c, r, g_f) :- at_idx(w, 11), at_idx(g, 24), at_idx(r, 45).
onface(wrg_c, r, y_f) :- at_idx(r, 11), at_idx(w, 24), at_idx(g, 45).
onface(wrg_c, w, o_f) :- at_idx(r, 11), at_idx(w, 24), at_idx(g, 45).
onface(wrg_c, g, g_f) :- at_idx(r, 11), at_idx(w, 24), at_idx(g, 45).
onface(wrg_c, r, y_f) :- at_idx(r, 11), at_idx(g, 24), at_idx(w, 45).
onface(wrg_c, g, o_f) :- at_idx(r, 11), at_idx(g, 24), at_idx(w, 45).
onface(wrg_c, w, g_f) :- at_idx(r, 11), at_idx(g, 24), at_idx(w, 45).
onface(wrg_c, g, y_f) :- at_idx(g, 11), at_idx(w, 24), at_idx(r, 45).
onface(wrg_c, w, o_f) :- at_idx(g, 11), at_idx(w, 24), at_idx(r, 45).
onface(wrg_c, r, g_f) :- at_idx(g, 11), at_idx(w, 24), at_idx(r, 45).
onface(wrg_c, g, y_f) :- at_idx(g, 11), at_idx(r, 24), at_idx(w, 45).
onface(wrg_c, r, o_f) :- at_idx(g, 11), at_idx(r, 24), at_idx(w, 45).
onface(wrg_c, w, g_f) :- at_idx(g, 11), at_idx(r, 24), at_idx(w, 45).
onface(wrb_c, w, y_f) :- at_idx(w, 11), at_idx(r, 24), at_idx(b, 45).
onface(wrb_c, r, o_f) :- at_idx(w, 11), at_idx(r, 24), at_idx(b, 45).
onface(wrb_c, b, g_f) :- at_idx(w, 11), at_idx(r, 24), at_idx(b, 45).
onface(wrb_c, w, y_f) :- at_idx(w, 11), at_idx(b, 24), at_idx(r, 45).
onface(wrb_c, b, o_f) :- at_idx(w, 11), at_idx(b, 24), at_idx(r, 45).
onface(wrb_c, r, g_f) :- at_idx(w, 11), at_idx(b, 24), at_idx(r, 45).
onface(wrb_c, r, y_f) :- at_idx(r, 11), at_idx(w, 24), at_idx(b, 45).
onface(wrb_c, w, o_f) :- at_idx(r, 11), at_idx(w, 24), at_idx(b, 45).
onface(wrb_c, b, g_f) :- at_idx(r, 11), at_idx(w, 24), at_idx(b, 45).
onface(wrb_c, r, y_f) :- at_idx(r, 11), at_idx(b, 24), at_idx(w, 45).
onface(wrb_c, b, o_f) :- at_idx(r, 11), at_idx(b, 24), at_idx(w, 45).
onface(wrb_c, w, g_f) :- at_idx(r, 11), at_idx(b, 24), at_idx(w, 45).
onface(wrb_c, b, y_f) :- at_idx(b, 11), at_idx(w, 24), at_idx(r, 45).
onface(wrb_c, w, o_f) :- at_idx(b, 11), at_idx(w, 24), at_idx(r, 45).
onface(wrb_c, r, g_f) :- at_idx(b, 11), at_idx(w, 24), at_idx(r, 45).
onface(wrb_c, b, y_f) :- at_idx(b, 11), at_idx(r, 24), at_idx(w, 45).
onface(wrb_c, r, o_f) :- at_idx(b, 11), at_idx(r, 24), at_idx(w, 45).
onface(wrb_c, w, g_f) :- at_idx(b, 11), at_idx(r, 24), at_idx(w, 45).
onface(yob_c, y, y_f) :- at_idx(y, 11), at_idx(o, 24), at_idx(b, 45).
onface(yob_c, o, o_f) :- at_idx(y, 11), at_idx(o, 24), at_idx(b, 45).
onface(yob_c, b, g_f) :- at_idx(y, 11), at_idx(o, 24), at_idx(b, 45).
onface(yob_c, y, y_f) :- at_idx(y, 11), at_idx(b, 24), at_idx(o, 45).
onface(yob_c, b, o_f) :- at_idx(y, 11), at_idx(b, 24), at_idx(o, 45).
onface(yob_c, o, g_f) :- at_idx(y, 11), at_idx(b, 24), at_idx(o, 45).
onface(yob_c, o, y_f) :- at_idx(o, 11), at_idx(y, 24), at_idx(b, 45).
onface(yob_c, y, o_f) :- at_idx(o, 11), at_idx(y, 24), at_idx(b, 45).
onface(yob_c, b, g_f) :- at_idx(o, 11), at_idx(y, 24), at_idx(b, 45).
onface(yob_c, o, y_f) :- at_idx(o, 11), at_idx(b, 24), at_idx(y, 45).
onface(yob_c, b, o_f) :- at_idx(o, 11), at_idx(b, 24), at_idx(y, 45).
onface(yob_c, y, g_f) :- at_idx(o, 11), at_idx(b, 24), at_idx(y, 45).
onface(yob_c, b, y_f) :- at_idx(b, 11), at_idx(y, 24), at_idx(o, 45).
onface(yob_c, y, o_f) :- at_idx(b, 11), at_idx(y, 24), at_idx(o, 45).
onface(yob_c, o, g_f) :- at_idx(b, 11), at_idx(y, 24), at_idx(o, 45).
onface(yob_c, b, y_f) :- at_idx(b, 11), at_idx(o, 24), at_idx(y, 45).
onface(yob_c, o, o_f) :- at_idx(b, 11), at_idx(o, 24), at_idx(y, 45).
onface(yob_c, y, g_f) :- at_idx(b, 11), at_idx(o, 24), at_idx(y, 45).
onface(yog_c, y, y_f) :- at_idx(y, 11), at_idx(o, 24), at_idx(g, 45).
onface(yog_c, o, o_f) :- at_idx(y, 11), at_idx(o, 24), at_idx(g, 45).
onface(yog_c, g, g_f) :- at_idx(y, 11), at_idx(o, 24), at_idx(g, 45).
onface(yog_c, y, y_f) :- at_idx(y, 11), at_idx(g, 24), at_idx(o, 45).
onface(yog_c, g, o_f) :- at_idx(y, 11), at_idx(g, 24), at_idx(o, 45).
onface(yog_c, o, g_f) :- at_idx(y, 11), at_idx(g, 24), at_idx(o, 45).
onface(yog_c, o, y_f) :- at_idx(o, 11), at_idx(y, 24), at_idx(g, 45).
onface(yog_c, y, o_f) :- at_idx(o, 11), at_idx(y, 24), at_idx(g, 45).
onface(yog_c, g, g_f) :- at_idx(o, 11), at_idx(y, 24), at_idx(g, 45).
onface(yog_c, o, y_f) :- at_idx(o, 11), at_idx(g, 24), at_idx(y, 45).
onface(yog_c, g, o_f) :- at_idx(o, 11), at_idx(g, 24), at_idx(y, 45).
onface(yog_c, y, g_f) :- at_idx(o, 11), at_idx(g, 24), at_idx(y, 45).
onface(yog_c, g, y_f) :- at_idx(g, 11), at_idx(y, 24), at_idx(o, 45).
onface(yog_c, y, o_f) :- at_idx(g, 11), at_idx(y, 24), at_idx(o, 45).
onface(yog_c, o, g_f) :- at_idx(g, 11), at_idx(y, 24), at_idx(o, 45).
onface(yog_c, g, y_f) :- at_idx(g, 11), at_idx(o, 24), at_idx(y, 45).
onface(yog_c, o, o_f) :- at_idx(g, 11), at_idx(o, 24), at_idx(y, 45).
onface(yog_c, y, g_f) :- at_idx(g, 11), at_idx(o, 24), at_idx(y, 45).
onface(yrb_c, y, y_f) :- at_idx(y, 11), at_idx(r, 24), at_idx(b, 45).
onface(yrb_c, r, o_f) :- at_idx(y, 11), at_idx(r, 24), at_idx(b, 45).
onface(yrb_c, b, g_f) :- at_idx(y, 11), at_idx(r, 24), at_idx(b, 45).
onface(yrb_c, y, y_f) :- at_idx(y, 11), at_idx(b, 24), at_idx(r, 45).
onface(yrb_c, b, o_f) :- at_idx(y, 11), at_idx(b, 24), at_idx(r, 45).
onface(yrb_c, r, g_f) :- at_idx(y, 11), at_idx(b, 24), at_idx(r, 45).
onface(yrb_c, r, y_f) :- at_idx(r, 11), at_idx(y, 24), at_idx(b, 45).
onface(yrb_c, y, o_f) :- at_idx(r, 11), at_idx(y, 24), at_idx(b, 45).
onface(yrb_c, b, g_f) :- at_idx(r, 11), at_idx(y, 24), at_idx(b, 45).
onface(yrb_c, r, y_f) :- at_idx(r, 11), at_idx(b, 24), at_idx(y, 45).
onface(yrb_c, b, o_f) :- at_idx(r, 11), at_idx(b, 24), at_idx(y, 45).
onface(yrb_c, y, g_f) :- at_idx(r, 11), at_idx(b, 24), at_idx(y, 45).
onface(yrb_c, b, y_f) :- at_idx(b, 11), at_idx(y, 24), at_idx(r, 45).
onface(yrb_c, y, o_f) :- at_idx(b, 11), at_idx(y, 24), at_idx(r, 45).
onface(yrb_c, r, g_f) :- at_idx(b, 11), at_idx(y, 24), at_idx(r, 45).
onface(yrb_c, b, y_f) :- at_idx(b, 11), at_idx(r, 24), at_idx(y, 45).
onface(yrb_c, r, o_f) :- at_idx(b, 11), at_idx(r, 24), at_idx(y, 45).
onface(yrb_c, y, g_f) :- at_idx(b, 11), at_idx(r, 24), at_idx(y, 45).
onface(yrg_c, y, y_f) :- at_idx(y, 11), at_idx(r, 24), at_idx(g, 45).
onface(yrg_c, r, o_f) :- at_idx(y, 11), at_idx(r, 24), at_idx(g, 45).
onface(yrg_c, g, g_f) :- at_idx(y, 11), at_idx(r, 24), at_idx(g, 45).
onface(yrg_c, y, y_f) :- at_idx(y, 11), at_idx(g, 24), at_idx(r, 45).
onface(yrg_c, g, o_f) :- at_idx(y, 11), at_idx(g, 24), at_idx(r, 45).
onface(yrg_c, r, g_f) :- at_idx(y, 11), at_idx(g, 24), at_idx(r, 45).
onface(yrg_c, r, y_f) :- at_idx(r, 11), at_idx(y, 24), at_idx(g, 45).
onface(yrg_c, y, o_f) :- at_idx(r, 11), at_idx(y, 24), at_idx(g, 45).
onface(yrg_c, g, g_f) :- at_idx(r, 11), at_idx(y, 24), at_idx(g, 45).
onface(yrg_c, r, y_f) :- at_idx(r, 11), at_idx(g, 24), at_idx(y, 45).
onface(yrg_c, g, o_f) :- at_idx(r, 11), at_idx(g, 24), at_idx(y, 45).
onface(yrg_c, y, g_f) :- at_idx(r, 11), at_idx(g, 24), at_idx(y, 45).
onface(yrg_c, g, y_f) :- at_idx(g, 11), at_idx(y, 24), at_idx(r, 45).
onface(yrg_c, y, o_f) :- at_idx(g, 11), at_idx(y, 24), at_idx(r, 45).
onface(yrg_c, r, g_f) :- at_idx(g, 11), at_idx(y, 24), at_idx(r, 45).
onface(yrg_c, g, y_f) :- at_idx(g, 11), at_idx(r, 24), at_idx(y, 45).
onface(yrg_c, r, o_f) :- at_idx(g, 11), at_idx(r, 24), at_idx(y, 45).
onface(yrg_c, y, g_f) :- at_idx(g, 11), at_idx(r, 24), at_idx(y, 45).
onface(wog_c, w, y_f) :- at_idx(w, 15), at_idx(o, 33), at_idx(g, 36).
onface(wog_c, o, r_f) :- at_idx(w, 15), at_idx(o, 33), at_idx(g, 36).
onface(wog_c, g, b_f) :- at_idx(w, 15), at_idx(o, 33), at_idx(g, 36).
onface(wog_c, w, y_f) :- at_idx(w, 15), at_idx(g, 33), at_idx(o, 36).
onface(wog_c, g, r_f) :- at_idx(w, 15), at_idx(g, 33), at_idx(o, 36).
onface(wog_c, o, b_f) :- at_idx(w, 15), at_idx(g, 33), at_idx(o, 36).
onface(wog_c, o, y_f) :- at_idx(o, 15), at_idx(w, 33), at_idx(g, 36).
onface(wog_c, w, r_f) :- at_idx(o, 15), at_idx(w, 33), at_idx(g, 36).
onface(wog_c, g, b_f) :- at_idx(o, 15), at_idx(w, 33), at_idx(g, 36).
onface(wog_c, o, y_f) :- at_idx(o, 15), at_idx(g, 33), at_idx(w, 36).
onface(wog_c, g, r_f) :- at_idx(o, 15), at_idx(g, 33), at_idx(w, 36).
onface(wog_c, w, b_f) :- at_idx(o, 15), at_idx(g, 33), at_idx(w, 36).
onface(wog_c, g, y_f) :- at_idx(g, 15), at_idx(w, 33), at_idx(o, 36).
onface(wog_c, w, r_f) :- at_idx(g, 15), at_idx(w, 33), at_idx(o, 36).
onface(wog_c, o, b_f) :- at_idx(g, 15), at_idx(w, 33), at_idx(o, 36).
onface(wog_c, g, y_f) :- at_idx(g, 15), at_idx(o, 33), at_idx(w, 36).
onface(wog_c, o, r_f) :- at_idx(g, 15), at_idx(o, 33), at_idx(w, 36).
onface(wog_c, w, b_f) :- at_idx(g, 15), at_idx(o, 33), at_idx(w, 36).
onface(wob_c, w, y_f) :- at_idx(w, 15), at_idx(o, 33), at_idx(b, 36).
onface(wob_c, o, r_f) :- at_idx(w, 15), at_idx(o, 33), at_idx(b, 36).
onface(wob_c, b, b_f) :- at_idx(w, 15), at_idx(o, 33), at_idx(b, 36).
onface(wob_c, w, y_f) :- at_idx(w, 15), at_idx(b, 33), at_idx(o, 36).
onface(wob_c, b, r_f) :- at_idx(w, 15), at_idx(b, 33), at_idx(o, 36).
onface(wob_c, o, b_f) :- at_idx(w, 15), at_idx(b, 33), at_idx(o, 36).
onface(wob_c, o, y_f) :- at_idx(o, 15), at_idx(w, 33), at_idx(b, 36).
onface(wob_c, w, r_f) :- at_idx(o, 15), at_idx(w, 33), at_idx(b, 36).
onface(wob_c, b, b_f) :- at_idx(o, 15), at_idx(w, 33), at_idx(b, 36).
onface(wob_c, o, y_f) :- at_idx(o, 15), at_idx(b, 33), at_idx(w, 36).
onface(wob_c, b, r_f) :- at_idx(o, 15), at_idx(b, 33), at_idx(w, 36).
onface(wob_c, w, b_f) :- at_idx(o, 15), at_idx(b, 33), at_idx(w, 36).
onface(wob_c, b, y_f) :- at_idx(b, 15), at_idx(w, 33), at_idx(o, 36).
onface(wob_c, w, r_f) :- at_idx(b, 15), at_idx(w, 33), at_idx(o, 36).
onface(wob_c, o, b_f) :- at_idx(b, 15), at_idx(w, 33), at_idx(o, 36).
onface(wob_c, b, y_f) :- at_idx(b, 15), at_idx(o, 33), at_idx(w, 36).
onface(wob_c, o, r_f) :- at_idx(b, 15), at_idx(o, 33), at_idx(w, 36).
onface(wob_c, w, b_f) :- at_idx(b, 15), at_idx(o, 33), at_idx(w, 36).
onface(wrg_c, w, y_f) :- at_idx(w, 15), at_idx(r, 33), at_idx(g, 36).
onface(wrg_c, r, r_f) :- at_idx(w, 15), at_idx(r, 33), at_idx(g, 36).
onface(wrg_c, g, b_f) :- at_idx(w, 15), at_idx(r, 33), at_idx(g, 36).
onface(wrg_c, w, y_f) :- at_idx(w, 15), at_idx(g, 33), at_idx(r, 36).
onface(wrg_c, g, r_f) :- at_idx(w, 15), at_idx(g, 33), at_idx(r, 36).
onface(wrg_c, r, b_f) :- at_idx(w, 15), at_idx(g, 33), at_idx(r, 36).
onface(wrg_c, r, y_f) :- at_idx(r, 15), at_idx(w, 33), at_idx(g, 36).
onface(wrg_c, w, r_f) :- at_idx(r, 15), at_idx(w, 33), at_idx(g, 36).
onface(wrg_c, g, b_f) :- at_idx(r, 15), at_idx(w, 33), at_idx(g, 36).
onface(wrg_c, r, y_f) :- at_idx(r, 15), at_idx(g, 33), at_idx(w, 36).
onface(wrg_c, g, r_f) :- at_idx(r, 15), at_idx(g, 33), at_idx(w, 36).
onface(wrg_c, w, b_f) :- at_idx(r, 15), at_idx(g, 33), at_idx(w, 36).
onface(wrg_c, g, y_f) :- at_idx(g, 15), at_idx(w, 33), at_idx(r, 36).
onface(wrg_c, w, r_f) :- at_idx(g, 15), at_idx(w, 33), at_idx(r, 36).
onface(wrg_c, r, b_f) :- at_idx(g, 15), at_idx(w, 33), at_idx(r, 36).
onface(wrg_c, g, y_f) :- at_idx(g, 15), at_idx(r, 33), at_idx(w, 36).
onface(wrg_c, r, r_f) :- at_idx(g, 15), at_idx(r, 33), at_idx(w, 36).
onface(wrg_c, w, b_f) :- at_idx(g, 15), at_idx(r, 33), at_idx(w, 36).
onface(wrb_c, w, y_f) :- at_idx(w, 15), at_idx(r, 33), at_idx(b, 36).
onface(wrb_c, r, r_f) :- at_idx(w, 15), at_idx(r, 33), at_idx(b, 36).
onface(wrb_c, b, b_f) :- at_idx(w, 15), at_idx(r, 33), at_idx(b, 36).
onface(wrb_c, w, y_f) :- at_idx(w, 15), at_idx(b, 33), at_idx(r, 36).
onface(wrb_c, b, r_f) :- at_idx(w, 15), at_idx(b, 33), at_idx(r, 36).
onface(wrb_c, r, b_f) :- at_idx(w, 15), at_idx(b, 33), at_idx(r, 36).
onface(wrb_c, r, y_f) :- at_idx(r, 15), at_idx(w, 33), at_idx(b, 36).
onface(wrb_c, w, r_f) :- at_idx(r, 15), at_idx(w, 33), at_idx(b, 36).
onface(wrb_c, b, b_f) :- at_idx(r, 15), at_idx(w, 33), at_idx(b, 36).
onface(wrb_c, r, y_f) :- at_idx(r, 15), at_idx(b, 33), at_idx(w, 36).
onface(wrb_c, b, r_f) :- at_idx(r, 15), at_idx(b, 33), at_idx(w, 36).
onface(wrb_c, w, b_f) :- at_idx(r, 15), at_idx(b, 33), at_idx(w, 36).
onface(wrb_c, b, y_f) :- at_idx(b, 15), at_idx(w, 33), at_idx(r, 36).
onface(wrb_c, w, r_f) :- at_idx(b, 15), at_idx(w, 33), at_idx(r, 36).
onface(wrb_c, r, b_f) :- at_idx(b, 15), at_idx(w, 33), at_idx(r, 36).
onface(wrb_c, b, y_f) :- at_idx(b, 15), at_idx(r, 33), at_idx(w, 36).
onface(wrb_c, r, r_f) :- at_idx(b, 15), at_idx(r, 33), at_idx(w, 36).
onface(wrb_c, w, b_f) :- at_idx(b, 15), at_idx(r, 33), at_idx(w, 36).
onface(yob_c, y, y_f) :- at_idx(y, 15), at_idx(o, 33), at_idx(b, 36).
onface(yob_c, o, r_f) :- at_idx(y, 15), at_idx(o, 33), at_idx(b, 36).
onface(yob_c, b, b_f) :- at_idx(y, 15), at_idx(o, 33), at_idx(b, 36).
onface(yob_c, y, y_f) :- at_idx(y, 15), at_idx(b, 33), at_idx(o, 36).
onface(yob_c, b, r_f) :- at_idx(y, 15), at_idx(b, 33), at_idx(o, 36).
onface(yob_c, o, b_f) :- at_idx(y, 15), at_idx(b, 33), at_idx(o, 36).
onface(yob_c, o, y_f) :- at_idx(o, 15), at_idx(y, 33), at_idx(b, 36).
onface(yob_c, y, r_f) :- at_idx(o, 15), at_idx(y, 33), at_idx(b, 36).
onface(yob_c, b, b_f) :- at_idx(o, 15), at_idx(y, 33), at_idx(b, 36).
onface(yob_c, o, y_f) :- at_idx(o, 15), at_idx(b, 33), at_idx(y, 36).
onface(yob_c, b, r_f) :- at_idx(o, 15), at_idx(b, 33), at_idx(y, 36).
onface(yob_c, y, b_f) :- at_idx(o, 15), at_idx(b, 33), at_idx(y, 36).
onface(yob_c, b, y_f) :- at_idx(b, 15), at_idx(y, 33), at_idx(o, 36).
onface(yob_c, y, r_f) :- at_idx(b, 15), at_idx(y, 33), at_idx(o, 36).
onface(yob_c, o, b_f) :- at_idx(b, 15), at_idx(y, 33), at_idx(o, 36).
onface(yob_c, b, y_f) :- at_idx(b, 15), at_idx(o, 33), at_idx(y, 36).
onface(yob_c, o, r_f) :- at_idx(b, 15), at_idx(o, 33), at_idx(y, 36).
onface(yob_c, y, b_f) :- at_idx(b, 15), at_idx(o, 33), at_idx(y, 36).
onface(yog_c, y, y_f) :- at_idx(y, 15), at_idx(o, 33), at_idx(g, 36).
onface(yog_c, o, r_f) :- at_idx(y, 15), at_idx(o, 33), at_idx(g, 36).
onface(yog_c, g, b_f) :- at_idx(y, 15), at_idx(o, 33), at_idx(g, 36).
onface(yog_c, y, y_f) :- at_idx(y, 15), at_idx(g, 33), at_idx(o, 36).
onface(yog_c, g, r_f) :- at_idx(y, 15), at_idx(g, 33), at_idx(o, 36).
onface(yog_c, o, b_f) :- at_idx(y, 15), at_idx(g, 33), at_idx(o, 36).
onface(yog_c, o, y_f) :- at_idx(o, 15), at_idx(y, 33), at_idx(g, 36).
onface(yog_c, y, r_f) :- at_idx(o, 15), at_idx(y, 33), at_idx(g, 36).
onface(yog_c, g, b_f) :- at_idx(o, 15), at_idx(y, 33), at_idx(g, 36).
onface(yog_c, o, y_f) :- at_idx(o, 15), at_idx(g, 33), at_idx(y, 36).
onface(yog_c, g, r_f) :- at_idx(o, 15), at_idx(g, 33), at_idx(y, 36).
onface(yog_c, y, b_f) :- at_idx(o, 15), at_idx(g, 33), at_idx(y, 36).
onface(yog_c, g, y_f) :- at_idx(g, 15), at_idx(y, 33), at_idx(o, 36).
onface(yog_c, y, r_f) :- at_idx(g, 15), at_idx(y, 33), at_idx(o, 36).
onface(yog_c, o, b_f) :- at_idx(g, 15), at_idx(y, 33), at_idx(o, 36).
onface(yog_c, g, y_f) :- at_idx(g, 15), at_idx(o, 33), at_idx(y, 36).
onface(yog_c, o, r_f) :- at_idx(g, 15), at_idx(o, 33), at_idx(y, 36).
onface(yog_c, y, b_f) :- at_idx(g, 15), at_idx(o, 33), at_idx(y, 36).
onface(yrb_c, y, y_f) :- at_idx(y, 15), at_idx(r, 33), at_idx(b, 36).
onface(yrb_c, r, r_f) :- at_idx(y, 15), at_idx(r, 33), at_idx(b, 36).
onface(yrb_c, b, b_f) :- at_idx(y, 15), at_idx(r, 33), at_idx(b, 36).
onface(yrb_c, y, y_f) :- at_idx(y, 15), at_idx(b, 33), at_idx(r, 36).
onface(yrb_c, b, r_f) :- at_idx(y, 15), at_idx(b, 33), at_idx(r, 36).
onface(yrb_c, r, b_f) :- at_idx(y, 15), at_idx(b, 33), at_idx(r, 36).
onface(yrb_c, r, y_f) :- at_idx(r, 15), at_idx(y, 33), at_idx(b, 36).
onface(yrb_c, y, r_f) :- at_idx(r, 15), at_idx(y, 33), at_idx(b, 36).
onface(yrb_c, b, b_f) :- at_idx(r, 15), at_idx(y, 33), at_idx(b, 36).
onface(yrb_c, r, y_f) :- at_idx(r, 15), at_idx(b, 33), at_idx(y, 36).
onface(yrb_c, b, r_f) :- at_idx(r, 15), at_idx(b, 33), at_idx(y, 36).
onface(yrb_c, y, b_f) :- at_idx(r, 15), at_idx(b, 33), at_idx(y, 36).
onface(yrb_c, b, y_f) :- at_idx(b, 15), at_idx(y, 33), at_idx(r, 36).
onface(yrb_c, y, r_f) :- at_idx(b, 15), at_idx(y, 33), at_idx(r, 36).
onface(yrb_c, r, b_f) :- at_idx(b, 15), at_idx(y, 33), at_idx(r, 36).
onface(yrb_c, b, y_f) :- at_idx(b, 15), at_idx(r, 33), at_idx(y, 36).
onface(yrb_c, r, r_f) :- at_idx(b, 15), at_idx(r, 33), at_idx(y, 36).
onface(yrb_c, y, b_f) :- at_idx(b, 15), at_idx(r, 33), at_idx(y, 36).
onface(yrg_c, y, y_f) :- at_idx(y, 15), at_idx(r, 33), at_idx(g, 36).
onface(yrg_c, r, r_f) :- at_idx(y, 15), at_idx(r, 33), at_idx(g, 36).
onface(yrg_c, g, b_f) :- at_idx(y, 15), at_idx(r, 33), at_idx(g, 36).
onface(yrg_c, y, y_f) :- at_idx(y, 15), at_idx(g, 33), at_idx(r, 36).
onface(yrg_c, g, r_f) :- at_idx(y, 15), at_idx(g, 33), at_idx(r, 36).
onface(yrg_c, r, b_f) :- at_idx(y, 15), at_idx(g, 33), at_idx(r, 36).
onface(yrg_c, r, y_f) :- at_idx(r, 15), at_idx(y, 33), at_idx(g, 36).
onface(yrg_c, y, r_f) :- at_idx(r, 15), at_idx(y, 33), at_idx(g, 36).
onface(yrg_c, g, b_f) :- at_idx(r, 15), at_idx(y, 33), at_idx(g, 36).
onface(yrg_c, r, y_f) :- at_idx(r, 15), at_idx(g, 33), at_idx(y, 36).
onface(yrg_c, g, r_f) :- at_idx(r, 15), at_idx(g, 33), at_idx(y, 36).
onface(yrg_c, y, b_f) :- at_idx(r, 15), at_idx(g, 33), at_idx(y, 36).
onface(yrg_c, g, y_f) :- at_idx(g, 15), at_idx(y, 33), at_idx(r, 36).
onface(yrg_c, y, r_f) :- at_idx(g, 15), at_idx(y, 33), at_idx(r, 36).
onface(yrg_c, r, b_f) :- at_idx(g, 15), at_idx(y, 33), at_idx(r, 36).
onface(yrg_c, g, y_f) :- at_idx(g, 15), at_idx(r, 33), at_idx(y, 36).
onface(yrg_c, r, r_f) :- at_idx(g, 15), at_idx(r, 33), at_idx(y, 36).
onface(yrg_c, y, b_f) :- at_idx(g, 15), at_idx(r, 33), at_idx(y, 36).
onface(wog_c, w, y_f) :- at_idx(w, 17), at_idx(o, 27), at_idx(g, 51).
onface(wog_c, o, r_f) :- at_idx(w, 17), at_idx(o, 27), at_idx(g, 51).
onface(wog_c, g, g_f) :- at_idx(w, 17), at_idx(o, 27), at_idx(g, 51).
onface(wog_c, w, y_f) :- at_idx(w, 17), at_idx(g, 27), at_idx(o, 51).
onface(wog_c, g, r_f) :- at_idx(w, 17), at_idx(g, 27), at_idx(o, 51).
onface(wog_c, o, g_f) :- at_idx(w, 17), at_idx(g, 27), at_idx(o, 51).
onface(wog_c, o, y_f) :- at_idx(o, 17), at_idx(w, 27), at_idx(g, 51).
onface(wog_c, w, r_f) :- at_idx(o, 17), at_idx(w, 27), at_idx(g, 51).
onface(wog_c, g, g_f) :- at_idx(o, 17), at_idx(w, 27), at_idx(g, 51).
onface(wog_c, o, y_f) :- at_idx(o, 17), at_idx(g, 27), at_idx(w, 51).
onface(wog_c, g, r_f) :- at_idx(o, 17), at_idx(g, 27), at_idx(w, 51).
onface(wog_c, w, g_f) :- at_idx(o, 17), at_idx(g, 27), at_idx(w, 51).
onface(wog_c, g, y_f) :- at_idx(g, 17), at_idx(w, 27), at_idx(o, 51).
onface(wog_c, w, r_f) :- at_idx(g, 17), at_idx(w, 27), at_idx(o, 51).
onface(wog_c, o, g_f) :- at_idx(g, 17), at_idx(w, 27), at_idx(o, 51).
onface(wog_c, g, y_f) :- at_idx(g, 17), at_idx(o, 27), at_idx(w, 51).
onface(wog_c, o, r_f) :- at_idx(g, 17), at_idx(o, 27), at_idx(w, 51).
onface(wog_c, w, g_f) :- at_idx(g, 17), at_idx(o, 27), at_idx(w, 51).
onface(wob_c, w, y_f) :- at_idx(w, 17), at_idx(o, 27), at_idx(b, 51).
onface(wob_c, o, r_f) :- at_idx(w, 17), at_idx(o, 27), at_idx(b, 51).
onface(wob_c, b, g_f) :- at_idx(w, 17), at_idx(o, 27), at_idx(b, 51).
onface(wob_c, w, y_f) :- at_idx(w, 17), at_idx(b, 27), at_idx(o, 51).
onface(wob_c, b, r_f) :- at_idx(w, 17), at_idx(b, 27), at_idx(o, 51).
onface(wob_c, o, g_f) :- at_idx(w, 17), at_idx(b, 27), at_idx(o, 51).
onface(wob_c, o, y_f) :- at_idx(o, 17), at_idx(w, 27), at_idx(b, 51).
onface(wob_c, w, r_f) :- at_idx(o, 17), at_idx(w, 27), at_idx(b, 51).
onface(wob_c, b, g_f) :- at_idx(o, 17), at_idx(w, 27), at_idx(b, 51).
onface(wob_c, o, y_f) :- at_idx(o, 17), at_idx(b, 27), at_idx(w, 51).
onface(wob_c, b, r_f) :- at_idx(o, 17), at_idx(b, 27), at_idx(w, 51).
onface(wob_c, w, g_f) :- at_idx(o, 17), at_idx(b, 27), at_idx(w, 51).
onface(wob_c, b, y_f) :- at_idx(b, 17), at_idx(w, 27), at_idx(o, 51).
onface(wob_c, w, r_f) :- at_idx(b, 17), at_idx(w, 27), at_idx(o, 51).
onface(wob_c, o, g_f) :- at_idx(b, 17), at_idx(w, 27), at_idx(o, 51).
onface(wob_c, b, y_f) :- at_idx(b, 17), at_idx(o, 27), at_idx(w, 51).
onface(wob_c, o, r_f) :- at_idx(b, 17), at_idx(o, 27), at_idx(w, 51).
onface(wob_c, w, g_f) :- at_idx(b, 17), at_idx(o, 27), at_idx(w, 51).
onface(wrg_c, w, y_f) :- at_idx(w, 17), at_idx(r, 27), at_idx(g, 51).
onface(wrg_c, r, r_f) :- at_idx(w, 17), at_idx(r, 27), at_idx(g, 51).
onface(wrg_c, g, g_f) :- at_idx(w, 17), at_idx(r, 27), at_idx(g, 51).
onface(wrg_c, w, y_f) :- at_idx(w, 17), at_idx(g, 27), at_idx(r, 51).
onface(wrg_c, g, r_f) :- at_idx(w, 17), at_idx(g, 27), at_idx(r, 51).
onface(wrg_c, r, g_f) :- at_idx(w, 17), at_idx(g, 27), at_idx(r, 51).
onface(wrg_c, r, y_f) :- at_idx(r, 17), at_idx(w, 27), at_idx(g, 51).
onface(wrg_c, w, r_f) :- at_idx(r, 17), at_idx(w, 27), at_idx(g, 51).
onface(wrg_c, g, g_f) :- at_idx(r, 17), at_idx(w, 27), at_idx(g, 51).
onface(wrg_c, r, y_f) :- at_idx(r, 17), at_idx(g, 27), at_idx(w, 51).
onface(wrg_c, g, r_f) :- at_idx(r, 17), at_idx(g, 27), at_idx(w, 51).
onface(wrg_c, w, g_f) :- at_idx(r, 17), at_idx(g, 27), at_idx(w, 51).
onface(wrg_c, g, y_f) :- at_idx(g, 17), at_idx(w, 27), at_idx(r, 51).
onface(wrg_c, w, r_f) :- at_idx(g, 17), at_idx(w, 27), at_idx(r, 51).
onface(wrg_c, r, g_f) :- at_idx(g, 17), at_idx(w, 27), at_idx(r, 51).
onface(wrg_c, g, y_f) :- at_idx(g, 17), at_idx(r, 27), at_idx(w, 51).
onface(wrg_c, r, r_f) :- at_idx(g, 17), at_idx(r, 27), at_idx(w, 51).
onface(wrg_c, w, g_f) :- at_idx(g, 17), at_idx(r, 27), at_idx(w, 51).
onface(wrb_c, w, y_f) :- at_idx(w, 17), at_idx(r, 27), at_idx(b, 51).
onface(wrb_c, r, r_f) :- at_idx(w, 17), at_idx(r, 27), at_idx(b, 51).
onface(wrb_c, b, g_f) :- at_idx(w, 17), at_idx(r, 27), at_idx(b, 51).
onface(wrb_c, w, y_f) :- at_idx(w, 17), at_idx(b, 27), at_idx(r, 51).
onface(wrb_c, b, r_f) :- at_idx(w, 17), at_idx(b, 27), at_idx(r, 51).
onface(wrb_c, r, g_f) :- at_idx(w, 17), at_idx(b, 27), at_idx(r, 51).
onface(wrb_c, r, y_f) :- at_idx(r, 17), at_idx(w, 27), at_idx(b, 51).
onface(wrb_c, w, r_f) :- at_idx(r, 17), at_idx(w, 27), at_idx(b, 51).
onface(wrb_c, b, g_f) :- at_idx(r, 17), at_idx(w, 27), at_idx(b, 51).
onface(wrb_c, r, y_f) :- at_idx(r, 17), at_idx(b, 27), at_idx(w, 51).
onface(wrb_c, b, r_f) :- at_idx(r, 17), at_idx(b, 27), at_idx(w, 51).
onface(wrb_c, w, g_f) :- at_idx(r, 17), at_idx(b, 27), at_idx(w, 51).
onface(wrb_c, b, y_f) :- at_idx(b, 17), at_idx(w, 27), at_idx(r, 51).
onface(wrb_c, w, r_f) :- at_idx(b, 17), at_idx(w, 27), at_idx(r, 51).
onface(wrb_c, r, g_f) :- at_idx(b, 17), at_idx(w, 27), at_idx(r, 51).
onface(wrb_c, b, y_f) :- at_idx(b, 17), at_idx(r, 27), at_idx(w, 51).
onface(wrb_c, r, r_f) :- at_idx(b, 17), at_idx(r, 27), at_idx(w, 51).
onface(wrb_c, w, g_f) :- at_idx(b, 17), at_idx(r, 27), at_idx(w, 51).
onface(yob_c, y, y_f) :- at_idx(y, 17), at_idx(o, 27), at_idx(b, 51).
onface(yob_c, o, r_f) :- at_idx(y, 17), at_idx(o, 27), at_idx(b, 51).
onface(yob_c, b, g_f) :- at_idx(y, 17), at_idx(o, 27), at_idx(b, 51).
onface(yob_c, y, y_f) :- at_idx(y, 17), at_idx(b, 27), at_idx(o, 51).
onface(yob_c, b, r_f) :- at_idx(y, 17), at_idx(b, 27), at_idx(o, 51).
onface(yob_c, o, g_f) :- at_idx(y, 17), at_idx(b, 27), at_idx(o, 51).
onface(yob_c, o, y_f) :- at_idx(o, 17), at_idx(y, 27), at_idx(b, 51).
onface(yob_c, y, r_f) :- at_idx(o, 17), at_idx(y, 27), at_idx(b, 51).
onface(yob_c, b, g_f) :- at_idx(o, 17), at_idx(y, 27), at_idx(b, 51).
onface(yob_c, o, y_f) :- at_idx(o, 17), at_idx(b, 27), at_idx(y, 51).
onface(yob_c, b, r_f) :- at_idx(o, 17), at_idx(b, 27), at_idx(y, 51).
onface(yob_c, y, g_f) :- at_idx(o, 17), at_idx(b, 27), at_idx(y, 51).
onface(yob_c, b, y_f) :- at_idx(b, 17), at_idx(y, 27), at_idx(o, 51).
onface(yob_c, y, r_f) :- at_idx(b, 17), at_idx(y, 27), at_idx(o, 51).
onface(yob_c, o, g_f) :- at_idx(b, 17), at_idx(y, 27), at_idx(o, 51).
onface(yob_c, b, y_f) :- at_idx(b, 17), at_idx(o, 27), at_idx(y, 51).
onface(yob_c, o, r_f) :- at_idx(b, 17), at_idx(o, 27), at_idx(y, 51).
onface(yob_c, y, g_f) :- at_idx(b, 17), at_idx(o, 27), at_idx(y, 51).
onface(yog_c, y, y_f) :- at_idx(y, 17), at_idx(o, 27), at_idx(g, 51).
onface(yog_c, o, r_f) :- at_idx(y, 17), at_idx(o, 27), at_idx(g, 51).
onface(yog_c, g, g_f) :- at_idx(y, 17), at_idx(o, 27), at_idx(g, 51).
onface(yog_c, y, y_f) :- at_idx(y, 17), at_idx(g, 27), at_idx(o, 51).
onface(yog_c, g, r_f) :- at_idx(y, 17), at_idx(g, 27), at_idx(o, 51).
onface(yog_c, o, g_f) :- at_idx(y, 17), at_idx(g, 27), at_idx(o, 51).
onface(yog_c, o, y_f) :- at_idx(o, 17), at_idx(y, 27), at_idx(g, 51).
onface(yog_c, y, r_f) :- at_idx(o, 17), at_idx(y, 27), at_idx(g, 51).
onface(yog_c, g, g_f) :- at_idx(o, 17), at_idx(y, 27), at_idx(g, 51).
onface(yog_c, o, y_f) :- at_idx(o, 17), at_idx(g, 27), at_idx(y, 51).
onface(yog_c, g, r_f) :- at_idx(o, 17), at_idx(g, 27), at_idx(y, 51).
onface(yog_c, y, g_f) :- at_idx(o, 17), at_idx(g, 27), at_idx(y, 51).
onface(yog_c, g, y_f) :- at_idx(g, 17), at_idx(y, 27), at_idx(o, 51).
onface(yog_c, y, r_f) :- at_idx(g, 17), at_idx(y, 27), at_idx(o, 51).
onface(yog_c, o, g_f) :- at_idx(g, 17), at_idx(y, 27), at_idx(o, 51).
onface(yog_c, g, y_f) :- at_idx(g, 17), at_idx(o, 27), at_idx(y, 51).
onface(yog_c, o, r_f) :- at_idx(g, 17), at_idx(o, 27), at_idx(y, 51).
onface(yog_c, y, g_f) :- at_idx(g, 17), at_idx(o, 27), at_idx(y, 51).
onface(yrb_c, y, y_f) :- at_idx(y, 17), at_idx(r, 27), at_idx(b, 51).
onface(yrb_c, r, r_f) :- at_idx(y, 17), at_idx(r, 27), at_idx(b, 51).
onface(yrb_c, b, g_f) :- at_idx(y, 17), at_idx(r, 27), at_idx(b, 51).
onface(yrb_c, y, y_f) :- at_idx(y, 17), at_idx(b, 27), at_idx(r, 51).
onface(yrb_c, b, r_f) :- at_idx(y, 17), at_idx(b, 27), at_idx(r, 51).
onface(yrb_c, r, g_f) :- at_idx(y, 17), at_idx(b, 27), at_idx(r, 51).
onface(yrb_c, r, y_f) :- at_idx(r, 17), at_idx(y, 27), at_idx(b, 51).
onface(yrb_c, y, r_f) :- at_idx(r, 17), at_idx(y, 27), at_idx(b, 51).
onface(yrb_c, b, g_f) :- at_idx(r, 17), at_idx(y, 27), at_idx(b, 51).
onface(yrb_c, r, y_f) :- at_idx(r, 17), at_idx(b, 27), at_idx(y, 51).
onface(yrb_c, b, r_f) :- at_idx(r, 17), at_idx(b, 27), at_idx(y, 51).
onface(yrb_c, y, g_f) :- at_idx(r, 17), at_idx(b, 27), at_idx(y, 51).
onface(yrb_c, b, y_f) :- at_idx(b, 17), at_idx(y, 27), at_idx(r, 51).
onface(yrb_c, y, r_f) :- at_idx(b, 17), at_idx(y, 27), at_idx(r, 51).
onface(yrb_c, r, g_f) :- at_idx(b, 17), at_idx(y, 27), at_idx(r, 51).
onface(yrb_c, b, y_f) :- at_idx(b, 17), at_idx(r, 27), at_idx(y, 51).
onface(yrb_c, r, r_f) :- at_idx(b, 17), at_idx(r, 27), at_idx(y, 51).
onface(yrb_c, y, g_f) :- at_idx(b, 17), at_idx(r, 27), at_idx(y, 51).
onface(yrg_c, y, y_f) :- at_idx(y, 17), at_idx(r, 27), at_idx(g, 51).
onface(yrg_c, r, r_f) :- at_idx(y, 17), at_idx(r, 27), at_idx(g, 51).
onface(yrg_c, g, g_f) :- at_idx(y, 17), at_idx(r, 27), at_idx(g, 51).
onface(yrg_c, y, y_f) :- at_idx(y, 17), at_idx(g, 27), at_idx(r, 51).
onface(yrg_c, g, r_f) :- at_idx(y, 17), at_idx(g, 27), at_idx(r, 51).
onface(yrg_c, r, g_f) :- at_idx(y, 17), at_idx(g, 27), at_idx(r, 51).
onface(yrg_c, r, y_f) :- at_idx(r, 17), at_idx(y, 27), at_idx(g, 51).
onface(yrg_c, y, r_f) :- at_idx(r, 17), at_idx(y, 27), at_idx(g, 51).
onface(yrg_c, g, g_f) :- at_idx(r, 17), at_idx(y, 27), at_idx(g, 51).
onface(yrg_c, r, y_f) :- at_idx(r, 17), at_idx(g, 27), at_idx(y, 51).
onface(yrg_c, g, r_f) :- at_idx(r, 17), at_idx(g, 27), at_idx(y, 51).
onface(yrg_c, y, g_f) :- at_idx(r, 17), at_idx(g, 27), at_idx(y, 51).
onface(yrg_c, g, y_f) :- at_idx(g, 17), at_idx(y, 27), at_idx(r, 51).
onface(yrg_c, y, r_f) :- at_idx(g, 17), at_idx(y, 27), at_idx(r, 51).
onface(yrg_c, r, g_f) :- at_idx(g, 17), at_idx(y, 27), at_idx(r, 51).
onface(yrg_c, g, y_f) :- at_idx(g, 17), at_idx(r, 27), at_idx(y, 51).
onface(yrg_c, r, r_f) :- at_idx(g, 17), at_idx(r, 27), at_idx(y, 51).
onface(yrg_c, y, g_f) :- at_idx(g, 17), at_idx(r, 27), at_idx(y, 51).
stk_f_dir(Cbl, StkCol, FTo, Dir) :- edge_cbl(Cbl), face_rel(FRef, FFrom, FTo, Dir), onface(Cbl, StkCol, FFrom), onface(Cbl, _, FRef).
% in_place.
in_place(Cbl) :- edge_cbl(Cbl), onface(Cbl, Col0, F0), face_col(F0, Col0), onface(Cbl, Col1, F1), face_col(F1, Col1), Col0 != Col1.
in_place(Cbl) :- corner_cbl(Cbl), onface(Cbl, Col0, F0), face_col(F0, Col0), onface(Cbl, Col1, F1), face_col(F1, Col1), onface(Cbl, Col2, F2), face_col(F2, Col2), Col0 != Col1, Col0 != Col2, Col1 != Col2.
in_place(Cbl) :- center_cbl(Cbl), onface(Cbl, Col0, F0), face_col(F0, Col0).
% constraints.
% different stickers from the same cubelet cannot be on the same face.
:- onface(Cbl, Col0, F), onface(Cbl, Col1, F), dif_col(Col0, Col1).
% cannot have a sticker color from same cubelet be on more than one face.
:- onface(Cbl, Col, F0), onface(Cbl, Col, F1), dif_face(F0, F1).
% cannot say a color of a cubelet is on a face if it does not have the color.
:- onface(Cbl, Col, _), not has_stk_col(Cbl, Col).
% cannot have two stickers from same cbl on opposite faces.
:- onface(Cbl, Col0, F0), onface(Cbl, Col1, F1), face_rel(_, F0, F1, op).
% cannot have two different colors at the same index.
:- at_idx(Col0, I), at_idx(Col1, I), dif_col(Col0, Col1).
#show at_idx/2.
#defined at_idx/2.
#defined center_cbl/1.
#defined edge_cbl/1.
#defined corner_cbl/1.
