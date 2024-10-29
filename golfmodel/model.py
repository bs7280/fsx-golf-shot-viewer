from scipy.integrate import odeint
from itertools import groupby
import pandas as pd
import numpy as np

class golf_ballstics():
    """
    Based on:
    
    ODEs taken from:
    ----------------   
    MacDonald and Hanzely (1991) "The physics of the drive in golf"
    American Journal of Physics 59(3):213-218
    Link: https://www.researchgate.net/publication/253220357_The_physics_of_the_drive_in_golf
    
    Lift and drag coefficients (Cf and Cl) as function of spin taken from:
    -----------------------------------
    A.J. Smiths, "A new aerodynamic model of a golf ball in flight"
    Science and Golf II: Proceedings of the World Scientific Congress on Golf, 1994
    https://www.researchgate.net/profile/Alexander_Smits/publication/284037213_A_new_aerodynamic_model_of_a_golf_ball_in_flight/links/5720f27708ae82260fab378b/A-new-aerodynamic-model-of-a-golf-ball-in-flight.pdf
    
    and 
    
    presentation
    B.D. Kothmann, "Aerodynamics of Sports Balls", January 2007
    http://www.seas.upenn.edu/~meam211/slides/aero.pdf
    based on http://people.stfx.ca/smackenz/courses/HK474/Labs/Jump%20Float%20Lab/Mehta%201985%20Aerodynamics%20of%20sports%20balls.pdf
    http://www.scielo.br/scielo.php?pid=S0102-47442004000400003&script=sci_arttext&tlng=en

    """
    
    def __init__(self, hyper_params = {}, verbose=False):
        self.verbose = verbose

        # golf ball properties
        self.mass = None
        self.radius = None
        
        # golf ball aerodynamic properties
        self.sn_Cl=[[0,0.04,0.1,0.2,0.4],
                    [0,0.1,0.16,0.23,0.33]]
        

        #cd=0.24+0.18*self.effective_spin(spin_rps, v)
        self.hyper_params = {}
        self.hyper_params['Cd_b'] = hyper_params.get('Cd_b', 0.24)
        self.hyper_params['Cd_m'] = hyper_params.get('Cd_m', 0.18)
        self.hyper_params['Cl_alpha'] = hyper_params.get('Cl_alpha', 0.52223327)
        
        # air properties
        self.rho = None
        
        # constants
        self.g = None
        
        # Initial ball flight properties at hit
        self.velocity = []
        self.spin = None
        self.spin_angle = None
        self.windvelocity = []
        
        # ODE solver parameters
        self.endtime = 10 # model ball flight for 10 sec
        self.timesteps = 100 # initial setting of time steps
        
        # For storing simulation results
        self.simres = None
        self.df_simres = pd.DataFrame(columns=['t', 'x', 'y', 'z', 'v_x', 'v_y', 'v_z'])

        
    def initiate_hit(self, velocity, launch_angle_deg, off_center_angle_deg, 
                     spin_rpm, spin_angle_deg, 
                     windspeed, windheading_deg,  
                     mass=0.0455, radius=0.0213, rho=1.225, g=9.81):
        """
        Objective:
        Solves ODE for time steps specified in the model (self.endtime and self.timesteps )
        Results are stored in self.df_simres
        
        Input:
        ------------------------------------------------------
        velocity              -   initial velocity  [m/s]
        launch_angle_deg      -   launch angle      [deg]
        off_center_angle_deg  -   aim               [deg]
        
        spin_rpm              -   ball spin         [rpm]
        spin_angle_deg        -   spin axis angle   [deg]
        
        windspeed             -   wind speed        [m/s] 
        windheading_deg       -   wind angle        [deg]
        
        Optional input:
        ------------------------------------------------------
        mass     -   mass of ball                  [kg]
        radius   -   radius of ball                [kg]
        rho      -   air density                   [kg/m^3]
        g        -   acceleration due to gravity   [kg]
        
        Output:
        ------------------------------------------------------
        Stored in pandas dataframe self.df_simres
        """
        
        # Set up initial parameters 
        self.mass = mass
        self.radius = radius
        self.rho = rho
        self.g = g
        
        self.spin_rps = spin_rpm/60 # revolutions per second
        self.spin_angle = spin_angle_deg/180*np.pi
        
        # Ball velocity vector
        theta = launch_angle_deg/180*np.pi
        psi = off_center_angle_deg/180*np.pi
        
        self.velocity = velocity*np.array([
            np.cos(theta)*np.sin(psi), # x 
            np.cos(theta)*np.cos(psi), # y
            np.sin(theta)              # z
        ])
        
        # Wind velocity vector
        windheading = windheading_deg/180*np.pi # 0 deg is tail wind (blowing with the hit for a straight shot)
        
        self.windvelocity = windspeed*np.array([
            np.sin(windheading), # x
            np.cos(windheading), # y
            0                    # z
        ]) 
        
        
        # Run simulation
        self.simulate()
    
    def get_landingpos(self, check = False, *args, **kwargs):
        """
        Wrapper of initiate_hit(). Returns coordinates (x, y) of ball when it hits the ground.
        If 'check = True' then a sanity check of the ball trajectory is performed and an additional
        string 'err' is returned.
        
        Input:
        ------------------------------------------------------
        args, kwargs   -   input to initiate_hit()
        check          -     

        Output: x, y OR x, y, err
        ------------------------------------------------------
        x    -   horizontal (left/right)
        y    -   length
        err  -   (Optional) error message, err = '' if everything is ok
        """

        imax = 3 # maximum number of model runs before giving up
        err = ''
        
        cont = True
        default_endtime = self.endtime
        i = 0
        
        while(cont):
            
            # Run model
            self.initiate_hit(*args, **kwargs)
            i += 1
            
            # Check output
            err = ''
            cont = False
            
            if(self.df_simres['z'][self.df_simres.shape[0]-1] > 0):
                err = 'error: ball never lands'
                
                # Try increasing analysis time
                self.endtime = self.endtime*2
                cont = True
                
            else:

                # Sanity checks
                ## -- Add/remove checks as needed --
                if check:
                    #if(np.nanmin(self.df_simres[self.df_simres['z']>=0]['x'].diff()) < 0):
                    #    err = 'error: ball moves backwards'

                    if(len(list(groupby(self.df_simres['z'], lambda x: x >= 0))) - 1 > 1):
                        err = 'error: ball passes through the ground multiple times'

            # Stop anyways
            if(i >= imax):
                cont = False
        
        self.endtime = default_endtime
        
        if(err == ''):
            # Linear interpolation to find where the ball lands

            index = np.argmax(np.array(self.df_simres['z']) < 0) - 1 # Last index where z > 0
            
            # Two points closest to z=0 surface
            p1 = (self.df_simres['x'][index], self.df_simres['y'][index], self.df_simres['z'][index])
            p2 = (self.df_simres['x'][index+1], self.df_simres['y'][index+1], self.df_simres['z'][index+1])
            
            # Intersection point
            t = p1[2]/(p1[2] - p2[2])
            x = p1[0] + t*(p2[0] - p1[0])
            y = p1[1] + t*(p2[1] - p1[1])
            
        else:
            x, y = 0, 0

        if check: return x, y, err
        return x, y
    
    def B(self):
        """ Constant B depending on ball and air properties """
        area = np.pi*self.radius**2
        return self.rho*area/(2*self.mass)
    
    def spin_rate_decay(self, t, initial_spin_rate):
        """
        Exponential decay model for spin rate.
        
        Parameters:
        - t: Time elapsed (in seconds).
        - initial_spin_rate: Initial spin rate of the golf ball.
        
        Returns:
        - spin_rate: Spin rate at time t.
        """
        decay_constant = 1 / 30  # Decay constant determining the rate of decay
        spin_rate = initial_spin_rate * np.exp(-t * decay_constant)
        return spin_rate

    def effective_spin(self, spin_rps, v):
        """ Effective spin used in calculation of drag and lift """
        #sn=self.spin*2*np.pi*self.radius/v
        sn=spin_rps*2*np.pi*self.radius/v
        return sn
        
    def Cd(self, v, spin_rps):        
        """ Drag coefficient """
        
        # From ??
        #cd=0.24+0.18*self.effective_spin(spin_rps, v)
        b = self.hyper_params['Cd_b'] #= 0.24
        m = self.hyper_params['Cd_m'] #= 0.18

        # Y = mx + b 
        # Where:
        #   y = Drag coefficient
        #   m (paper set to 0.18)
        #   b (paper set to 0.24)
        #   x : Effective spin
        cd=b+m*self.effective_spin(spin_rps, v)
        
        # From MacDonald and Hanzely (Based on Erlicson paper)
        #cd = 0.3048*46/v

        return cd
        
    def coef_lift(self, spin_rps, v,):
        a = self.hyper_params['Cl_alpha']
        b = 0
        s = self.effective_spin(spin_rps, v)
        return a * np.sqrt(s) + b
        
    def Cl(self, v, spin_rps):
        """ Lift coefficient """
        
        # From A.J. Smiths (1994)
        #cl=np.interp(x=self.effective_spin(spin_rps, v), xp=self.sn_Cl[0], fp=self.sn_Cl[1])
        
        cl = self.coef_lift(spin_rps, v)

        # From MacDonald and Hanzely (Based on Erlicson paper)
        #cl = 0.3048*33.4/v

        return cl
        
        
    def model(self,state,t):
        """ The ODE equations for ball velocity """
        
        # Current state at time step t
        x,y,z,vx,vy,vz = state
        
        # Calculate total velocity w.r.t. wind
        v_ball = [vx, vy, vz] # Ball velocity
        v_wind = self.windvelocity # Wind velocity 
        v_rel = v_ball - v_wind # Relative velocity to air
        u = np.linalg.norm(v_rel)
        
        # ODE coefficients
        a = self.spin_angle
        B = self.B()

        # Adding spin decay
        spin_t = self.spin_rate_decay(t, self.spin_rps)

        Cl = self.Cl(u, spin_t)
        Cd = self.Cd(u, spin_t)

        if self.verbose:
            print(f"t: {t:,.4f} vx: {vx:,.2f} vy: {vy:.2f}, v_rel: {u:,.3f} --- Cd: {Cd:,.3f} Cl: {Cl:,.3f}")
        #print("Cl", Cl)
        #print("Cd", Cd)
        
        # ODE equations
        ux, uy, uz = v_rel
        dvxdt = -B*u*(Cd*ux + Cl*uy*np.sin(a))
        dvydt = -B*u*(Cd*uy - Cl*(ux*np.sin(a) - uz*np.cos(a)))
        dvzdt = -self.g - B*u*(Cd*uz - Cl*uy*np.cos(a))
     
        return [vx, vy, vz, dvxdt, dvydt, dvzdt]
        
        
    def simulate(self):
        """ Run simulation """
        
        # Time steps
        self.df_simres['t'] = np.linspace(0, self.endtime, self.timesteps)
        
        # Initial state 
        v0 = [0, 0, 0] + self.velocity.tolist()
        
        # Solve ODE
        self.simres = odeint(self.model,v0,self.df_simres['t'])
        
        # Store results in dataframe
        self.df_simres['x']= self.simres[:,0] 
        self.df_simres['y']= self.simres[:,1] 
        self.df_simres['z']= self.simres[:,2]
        self.df_simres['v_x']= self.simres[:,3] 
        self.df_simres['v_y']= self.simres[:,4] 
        self.df_simres['v_z']= self.simres[:,5]
        
if __name__ == '__main__':
    mph_to_ms = 0.44704 # 1 mph = 0.447m/s
    meters_to_yards = 1.09361

    def cal_golf_landing_mph(
        velocity=145,
        launch_angle_deg=10,
        off_center_angle_deg=-2.3,
        spin_rpm=3300,
        spin_angle_deg=3.1,
        windspeed=0,
        windheading_deg=100):

        golf_m = golf_ballstics(verbose=False, hyper_params={
            'Cd_b': 0.23, 'Cl_alpha': 0.52223327 * 1.1})

        y_distance, x_distance = golf_m.get_landingpos(
            velocity=velocity*mph_to_ms, 
            launch_angle_deg=launch_angle_deg,
            off_center_angle_deg=off_center_angle_deg, 
            spin_rpm=spin_rpm,
            spin_angle_deg=-1 * spin_angle_deg,
            windspeed=windspeed * mph_to_ms,
            windheading_deg=windheading_deg)

        # Convert ouput to Yards from Meters
        y_yards = meters_to_yards * y_distance
        x_yards = meters_to_yards * x_distance
        return x_yards, y_yards
        #print('Ball lands at ({},{})'.format(y_yards, x_yards))


    landing = cal_golf_landing_mph()#off_center_angle_deg=-2, spin_angle_deg=0)

    print(f"Landing: {landing}")