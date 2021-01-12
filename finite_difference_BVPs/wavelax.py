import numpy as np
import matplotlib.pyplot as mpl


#fixed parameters
c = 300
L = 1.0
x0 = 0.3
s = 0.02



#input parameters
dx = L*float(input('grid spacing in units of wire lenght (L) ->'))
dt = dx/c*float(input('time step in units of (dx/c) ->'))
tmax = L/c*float(input('evolution time in units of (L/c) ->'))

#construct initial data
N = int(L/dx)
x = [0.0]*(N+1)
u0 =[0.0]*(N+1)
v0 = [0.0]*(N+1)
u1 =[0.0]*(N+1)
v1 =[0.0]*(N+1)

for j in range(N+1):
   x[j] = j*dx
   u0[j] = np.exp(-0.5*((x[j]-x0/s)**2))

#prepare animated plot
mpl.ion()
(line,)=mpl.plot(x,u0,'-k')
mpl.ylim(-1.2,1.2)
mpl.xlabel('x(m)')
mpl.ylabel('u')

#perform evolution
t = 0.0
while t < tmax:
     #update plot
     line.set_ydata(u0)
     mpl.title('t=%5f'%t)
     mpl.draw()
     mpl.pause(0.1)
     #derivatives at interior points
     for j in range(1,N):
       v1[j]=0.5*(v0[j-1]+v0[j+1])+0.5*dt*c*(u0[j+1]-u0[j-1])/dx
       u1[j]=0.5*(u0[j+1]+u0[j+1])+0.5*dt*c*(v0[j+1]-u0[j-1])/dx

     #boundary conditions
     u1[0]=u1[N]=0.0
     v1[0]=v1[1]+u1[1]
     v1[N]=v1[N-1]-u1[N-1]

     #swap old and new lists
     (u0,u1)=(u1,u0)
     (v0,v1) = (v1,v0)
     t += dt

#freeze final plot
mpl.ioff()
mpl.show()





