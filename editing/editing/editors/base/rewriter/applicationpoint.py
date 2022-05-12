from abc import ABC


ApplicationPoint = type('ApplicationPoint', (ABC,), {})  # in this way, we let each `Rewriter` define what an `ApplicationPoint` is for it
