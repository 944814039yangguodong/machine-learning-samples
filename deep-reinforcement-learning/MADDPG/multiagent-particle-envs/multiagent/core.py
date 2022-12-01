import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None
        # angular
        self.p_angular = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property 
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()
        # physical motor noise amount
        self.u_noise = None
        # action
        self.action = Action()
        # flag
        self.isLandmark = True

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # flag
        self.isLandmark = False

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        self.time = 0

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]



    # update state of the world
    def step(self):
        self.time += self.dt
        # 一致性算法更新地标位置
        # self.update_landmark()
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,entity in enumerate(self.entities):
            if entity.movable:
                noise = np.random.randn(*entity.action.u.shape) * entity.u_noise if entity.u_noise else 0.0
                p_force[i] = entity.action.u + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        # print("总体环境碰撞力:"+str(p_force))
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping * 0)
            # if(i==0):
            #     print("实体"+str(i)+"原速度："+str(entity.state.p_vel))
            #     print("实体"+str(i)+"原位置："+str(entity.state.p_pos))
            #     print("实体"+str(i)+"动作:"+str(p_force[i]))
            #     print("实体"+str(i)+"时间步长:"+str(self.dt))
            if (p_force[i] is not None):
                entity.state.p_vel += p_force[i]*self.dt
            if entity.max_speed is not None:
                if entity.state.p_vel[0] > entity.max_speed:
                    entity.state.p_vel[0] = entity.max_speed
                if entity.state.p_vel[1] > entity.max_speed:
                    entity.state.p_vel[1] = entity.max_speed
            # 现位置
            entity.state.p_pos[0] += entity.state.p_vel[0]*self.dt + 0.5*p_force[i][0]*self.dt*self.dt
            entity.state.p_pos[1] += entity.state.p_vel[1]*self.dt + 0.5*p_force[i][1]*self.dt*self.dt
            # if(i==0):
            #     print("实体"+str(i)+"现速度："+str(entity.state.p_vel))
            #     print("实体"+str(i)+"现位置："+str(entity.state.p_pos))

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    def update_landmark(self):
        M = 4 # agent数量
        n = 2 # 空间维度
        a = np.zeros(2*n*M)
        for i,Landmark in enumerate(self.landmarks):
            if i<4:
                a[n*i:n*i+2] = Landmark.state.p_pos
                a[n*M+n*i:n*M+n*i+2] = Landmark.state.p_vel
        tmp = self.paper_discrete(a,a,self.dt)
        for i,Landmark in enumerate(self.landmarks):
            if i<4:
                Landmark.state.p_pos[0]=tmp[n*i]
                Landmark.state.p_pos[1]=tmp[n*i+1]
                Landmark.state.p_vel[0]=tmp[n*M+n*i]
                Landmark.state.p_vel[1]=tmp[n*M+n*i+1]
                #print("更新地标"+str(i)+":"+str(Landmark.state.p_pos))

    def paper_discrete(self,y1,y2,space):
        M = 4 # agent数量
        n = 2 # 空间维度
        c1 = 5 # 目标速度系数
        c2 = 2 # 队形加速度系数
        # 4个agent的目标速度
        vd = 0*np.ones((n,M)) 
        if self.time >= 5:
            vd = 0.03*np.ones((n,M)) 
        # 4个agent的目标队形
        xd = np.array([[1/2,1/2],
            [0,1/2],
            [0,0],
            [1/2,0]]) 
        if self.time >= 10:
            xd = np.array([[1,1],
            [0,1],
            [0,0],
            [1/2,1/2]])  
        xd = np.array(list(map(list, zip(*xd))))
        
        # 4个agent的通信拓扑图邻接矩阵
        A = np.array([[0,1,1,1],
            [1,0,1,1],
            [1,1,0,1],
            [1,1,1,0]])
        xm = np.zeros((n, M))
        vm = np.zeros((n, M))
        xlag1 = np.zeros((n, M))
        vlag1 = np.zeros((n, M))
        xdot = np.zeros((n, M))
        vdot = np.zeros((n, M))
        out = np.zeros((2*n*M,1))

        for i in range(0,M):
            xm[:, i] = y1[n*i:n*i+2]
            vm[:, i] = y1[n*M+n*i:n*M+n*i+2]

        for i in range(0,M):
            xlag1[:, i] = y2[n*i:n*i+2]
            vlag1[:, i] = y2[n*M+n*i:n*M+n*i+2]

        for i in range(0,M):
            xdot[:,i] = xm[:,i]+vm[:,i]*space-c1*(vm[:,i]-vd[:,i])*space**2/2
            vdot[:,i] = vm[:,i]-c1*(vm[:,i]-vd[:,i])*space
            for j in range(0,M):
                if j != i:
                    xdot[:,i]=xdot[:,i]+c2*A[i,j]*(xlag1[:,j]-xd[:,j]-xlag1[:,i]+xd[:,i])*space**2/2
                    vdot[:,i]=vdot[:,i]+c2*A[i,j]*(xlag1[:,j]-xd[:,j]-xlag1[:,i]+xd[:,i])*space

        for i in range(0,M):
            tmp = np.zeros((n,1))
            for j in range(0,n):
                tmp[j][0]=xdot[j,i]
            tmp2= np.zeros((n,1))
            for j in range(0,n):
                tmp2[j][0]=vdot[0,i]
            out[n*i:n*i+2] = tmp
            out[n*M+n*i:n*M+n*i+2] = tmp2
        return out