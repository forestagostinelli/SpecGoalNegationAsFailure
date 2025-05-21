% not_at_row :- not at_col(t8, c0).
% row_alleven(R) :- row(R), num_row(NRow), #count{ T: at_row(T,R), val(T,V), V \ 2 = 0} = NRow.
% row_hasodd(R) :- row(R), #count{ T: at_row(T,R), val(T,V), V \ 2 = 1} > 0.
% rowsum_lte(R,Th) :- row(R), #sum{ V: at_row(T,R), val(T,V) } = S, num(Th), S <= Th.
% rowsum_odd(R) :- row(R), rowsum(R, S), S \ 2 = 1.
% exist_rowsum_odd :- row(R), rowsum_odd(R).

even(V) :- num(V), V \ 2 = 0.
odd(V) :- num(V), V \ 2 = 1.

rowsum(R,S) :- row(R), num_row(NRow), #count{ T: at_row(T,R) } = NRow, #sum{ V: at_row(T,R), val(T,V) } = S.

rowsum_even(R) :- rowsum(R,S), even(S).
rowsum_odd(R) :- rowsum(R,S), odd(S).

rowsum_even_any :- row(R), rowsum_even(R).
rowsum_odd_any :- row(R), rowsum_odd(R).

rowsum_even_all :- num_row(NRow), #count{ R: rowsum_even(R) } = NRow.
rowsum_odd_all :- num_row(NRow), #count{ R: rowsum_odd(R) } = NRow.

r0_sum_even :- rowsum_even(r0).
not_r0_sum_odd :- not rowsum_odd(r0).
all_r_sum_even :- rowsum_even_all.
not_any_r_sum_odd :- not rowsum_odd_any.