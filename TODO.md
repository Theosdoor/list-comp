# TODO

## main

- Make a seperate repo for paper / thesis
- seperate scripts into sweeps, model, sae folders. also train scripts shouldnt be nbs


## Extensions

- lots of saes have 2 special features. would be good to try scaling them together rather than indep and see if we can swap even more ouputs
  - could maybe scale up to 90% of inputs because 90% of inputs activate one of the (never both) special latents
  - alterntatively, if we get no o2 crossover in bounds for 1 special feat eg. k=3 then we can try scaling other latents in grid searchand see if we can get swap zone (usually no o2 in bounds means it needs -ve scale, and can sometimes scale other active latents to move the crossover into +ve scale region). The plus of this is that k=3 is much easier to do a grid search on than k>3 ! and maybe it's just as good
- 
