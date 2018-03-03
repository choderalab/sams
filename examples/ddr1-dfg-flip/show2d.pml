load 2d-distance-trajectory.pdb
hide all
color green, all
load_traj 2d-distance-trajectory.xtc
intra_fit all
show cartoon, all
select sidechains, not hydrogen and (resi 51 or resi 153 or resi 180 or resi 181)
show sticks, sidechains
set cartoon_side_chain_helper, 1
util.cbay('sidechains')
dist resi 153 and name CB, resi 181 and name CZ
dist resi 51 and name NZ, resi 180 and name CG
select dihedral, (resi 180 and name CA) or (resi 180 and name N) or (resi 181 and name N) or (resi 181 and name CG)
color red, dihedral