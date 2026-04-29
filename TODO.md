# TODO

## Extensions

- speed up xover pipeline by skipping inputs that don't activate the special feature - eg. d1=d2
- lots of saes have 2 special features. would be good to try scaling them together rather than indep and see if we can swap even more ouputs
  - could maybe scale up to 90% of inputs because 90% of inputs activate one of the (never both) special latents
  - alterntatively, if we get no o2 crossover in bounds for 1 special feat eg. k=3 then we can try scaling other latents in grid searchand see if we can get swap zone (usually no o2 in bounds means it needs -ve scale, and can sometimes scale other active latents to move the crossover into +ve scale region). The plus of this is that k=3 is much easier to do a grid search on than k>3 ! and maybe it's just as good
  - NOTE - doesnt make sense to scale both special feats at same time because they never co-activate. So only option is to scale all other activate latents for each eg. with no o2 xover in bounds ==> need max (# scale steps)^k lots of steps in xover analysis
    - can go in order of strongest - weakest activation, or ignore latents that don't alter pce when ablated

