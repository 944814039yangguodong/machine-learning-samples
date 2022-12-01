import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 4
        num_landmarks = 4
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = '%d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.025
            # agent.max_speed = 5
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = '%d' % i
            landmark.collide = False
            landmark.movable = True
            landmark.size = 0.015
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.time = 0
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.15*i, 0.25*i, 0.15*i])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15*i, 0.25*i, 0.15*i])
        # set random initial states
        pos = np.zeros((4,2))
        for i,agent in enumerate(world.agents):
            agent.state.p_vel = np.array([0,0])
            agent.state.c = np.array([0,0])
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            pos[i]=agent.state.p_pos
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_vel = np.array([0,0])
            landmark.action.u=np.array([0,0])
            #landmark.state.p_pos = pos[i]+np.random.uniform(-0.001, +0.001, world.dim_p)
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent) and agent.name!=a.name:
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def almost_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = 2*(agent1.size + agent2.size)
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for i, l in enumerate(world.landmarks):
            if str(i)==agent.name:
                rew -= np.sqrt(np.square((agent.state.p_pos - l.state.p_pos)[0])+np.square((agent.state.p_pos - l.state.p_pos)[1]))
        if agent.collide:
            for a in world.agents:
                if self.almost_collision(a, agent) and agent.name!=a.name:
                    rew -= 0.16/np.sqrt(np.square((agent.state.p_pos - a.state.p_pos)[0])+np.square((agent.state.p_pos - a.state.p_pos)[1]))
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            if(agent.name==entity.name):
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        tmp=np.concatenate([agent.state.p_pos]+ [agent.state.p_vel] + entity_pos + other_pos)
        return tmp