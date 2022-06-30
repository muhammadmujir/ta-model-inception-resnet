# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:52:11 2022

@author: Mujir
"""
import matplotlib.pyplot as plt
import numpy as np

def main():
    painting=plt.imread("https://www.freepnglogos.com/uploads/server-png/home-server-icon-icons-and-png-backgrounds-30.png")
    for i in range(4):
        print(i)
        plt.figure()
        plt.imshow(painting)
      
if __name__ == '__main__' :
    main()