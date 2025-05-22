%tiles and blanks.
tile(t1).
tile(t2).
tile(t3).
tile(t4).
tile(t5).
tile(t6).
tile(t7).
tile(t8).
tile(t9).
tile(t10).
tile(t11).
tile(t12).
tile(t13).
tile(t14).
tile(t15).
tile(t16).
tile(t17).
tile(t18).
tile(t19).
tile(t20).
tile(t21).
tile(t22).
tile(t23).
tile(t24).
blank(t0).
t_or_b(X) :- tile(X).
t_or_b(X) :- blank(X).
val(t0, 0).
val(t1, 1).
val(t2, 2).
val(t3, 3).
val(t4, 4).
val(t5, 5).
val(t6, 6).
val(t7, 7).
val(t8, 8).
val(t9, 9).
val(t10, 10).
val(t11, 11).
val(t12, 12).
val(t13, 13).
val(t14, 14).
val(t15, 15).
val(t16, 16).
val(t17, 17).
val(t18, 18).
val(t19, 19).
val(t20, 20).
val(t21, 21).
val(t22, 22).
val(t23, 23).
val(t24, 24).
num(0..325).

%rows and columns.
row(r0).
row(r1).
row(r2).
row(r3).
row(r4).
col(c0).
col(c1).
col(c2).
col(c3).
col(c4).
at_row(X, R) :- t_or_b(X), row(R), at_idx(X, R, _).
at_col(X, C) :- t_or_b(X), col(C), at_idx(X, _, C).
num_row(5).
num_col(5).

% classical negation.
-at_idx(X, R, C) :- t_or_b(X), t_or_b(X2), at_idx(X2, R, C), not X=X2.
-at_idx(X, R, C) :- row(R), col(C), row(R2), at_idx(X, R2, _), not R=R2.
-at_idx(X, R, C) :- row(R), col(C), row(C2), at_idx(X, _, C2), not C=C2.

% constraints.
% location cannot have multiple tiles.
:- row(R), col(C), #count{X: at_idx(X, R, C)} > 1.
% tile or blank cannot be in more than one place at a time.
:- t_or_b(X), #count{R, C: at_idx(X, R, C)} > 1.
