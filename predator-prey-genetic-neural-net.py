# To DO
###############################################
# Fix Num Organism Counter (Broken)
# Split classes into multiple files
# Ability to save best performing neural nets
# Add plants for the prey to eat
# Reference https://www.youtube.com/watch?v=qwrp3lB-jkQ
###############################################


import pygame
import pygame_gui
import numpy as np
import random
import copy
# Constants

window_size = (800, 600)
frame_rate=24

organism_radius=5
num_organisms_start=100


max_energy=100
max_angular_velocity=20
max_linear_velocity=10
num_sensor_lines=30

motion_energy_loss_constant=0.2

max_prey=200
prey_energy_regen_at_rest=90
prey_split_time=10
prey_sense_range=100
prey_sense_angle=330

max_predator=50
predator_energy_degen=0.4
predator_energy_gain_on_eat=100
predator_split_eat_amount=2
predator_eat_cooldown=0.1
predator_sense_range=500
predator_sense_angle=180

mutation_rate=0.2

# num_prey=0
# num_predator=0

class Organism:
    def __init__(self, position, neuralnet):
        self.isAlive=True
        self.position=position
        self.neuralnet=neuralnet
        self.angle=0
        self.linear_velocity=0
        self.angular_velocity=0
        self.energy=max_energy
        self.sense_range=0
        self.sense_angle=0
        self.color=(0,0,0)
        self.is_alive=True


    
    def iterate(self):
        thought = self.think()
        self.linear_velocity=thought[0]
        self.angular_velocity=thought[1]
        if (self.linear_velocity>max_linear_velocity):
            self.linear_velocity=max_linear_velocity
        if (self.angular_velocity>max_angular_velocity):
            self.angular_velocity=max_angular_velocity
        self.energy=self.energy-motion_energy_loss_constant*np.sqrt((np.abs(self.linear_velocity)))
        self.position[0]+=self.linear_velocity*np.cos(np.deg2rad(self.angle))
        self.position[1]-=self.linear_velocity*np.sin(np.deg2rad(self.angle))
        self.angle+=self.angular_velocity
        if self.position[0] - organism_radius< 0:
            self.position[0] = window_size[0] - organism_radius
        elif self.position[0] + organism_radius > window_size[0]:
            self.position[0] =  organism_radius

        if self.position[1] - organism_radius < 0:
            self.position[1] = window_size[1] - organism_radius
        elif self.position[1] + organism_radius > window_size[1]:
            self.position[1] = organism_radius
        return
    
    def sense(self):
        senses=np.zeros(num_sensor_lines)

        angles=np.linspace(self.angle-self.sense_angle/2, self.angle+self.sense_angle/2,num_sensor_lines).tolist()
        for organism in organisms:
            if self.is_prey() != organism.is_prey():
                angle_to_organism = (np.degrees(np.arctan2(-organism.position[1] +self.position[1], organism.position[0] - self.position[0])) + 360) % 360
                distance_to_organism = ((organism.position[0] - self.position[0]) ** 2 + (-organism.position[1] +self.position[1]) ** 2) ** 0.5
                if distance_to_organism!=0 and distance_to_organism<self.sense_range and angle_to_organism<=max(angles) and angle_to_organism>=min(angles):
                    closest_angle_index = np.argmin(np.abs(np.array(angles) - angle_to_organism))
                    if senses[closest_angle_index]==0 or distance_to_organism<senses[closest_angle_index]:
                        senses[closest_angle_index]=1/distance_to_organism
        max_val = max(senses)
        min_val = min(senses)
        
        if max_val == min_val:
            return [0.0] * len(senses)
        
        senses = [(2 * (x - min_val) / (max_val - min_val)) - 1 for x in senses]
        return senses

    def think(self):
        return self.neuralnet.calculate(self.sense())
    
    def draw(self):
        pygame.draw.circle(window, self.color, self.position, organism_radius)
        return

    def split(self):
        return

    def death(self):
        self.is_alive=False

# Prey
# Energy Gain: Still
# Energy Loss: Motion
# Death: Eaten
# Split: Timer

class Prey(Organism):
    num_prey=0
    def __init__(self, position, neuralnet):
        super().__init__(position, neuralnet)
        Prey.num_prey+=1
        if Prey.num_prey>max_prey:
            self.death()
        self.sense_range=prey_sense_range
        self.sense_angle=prey_sense_angle
        self.color=(0, 255, 255)
        self.split_timer=0
        self.is_resting=False
    def is_prey(self):
        return True
    def iterate(self):
        super().iterate()
        if self.split_timer<prey_split_time:
            self.split_timer+=1/frame_rate
        else:
            self.split_timer=0
            self.split()
        if self.energy<=0:
            self.energy=0
            self.is_resting=True
        elif self.energy >=max_energy:
            self.energy=max_energy
            self.is_resting=False
        if self.is_resting:
            self.energy+=prey_energy_regen_at_rest/frame_rate
            self.position[0]-=self.linear_velocity*np.cos(np.deg2rad(self.angle))
            self.position[1]+=self.linear_velocity*np.sin(np.deg2rad(self.angle))
            self.angle-=self.angular_velocity
            if self.position[0] - organism_radius< 0:
                self.position[0] = window_size[0] - organism_radius
            elif self.position[0] + organism_radius > window_size[0]:
                self.position[0] =  organism_radius

            if self.position[1] - organism_radius < 0:
                self.position[1] = window_size[1] - organism_radius
            elif self.position[1] + organism_radius > window_size[1]:
                self.position[1] = organism_radius

    def split(self):
        super().split()
        new_neuralnet = self.neuralnet.mutate()
        organisms.append(Prey(self.position+np.random.uniform(-2*organism_radius, 2*organism_radius, size=2), new_neuralnet))
    def death(self):
        super().death()
        Prey.num_prey-=1
        if (Prey.num_prey==0):
            neural_net = NeuralNetwork(num_sensor_lines, 16, 2, 2)
            organisms.append(Prey([random.random()*window_size[0],random.random()*window_size[1]], neural_net))

# Predator
# Energy Gain: Eat (Cooldown)
# Energy Loss: Existence, Motion
# Death: No Energy
# Split: Twice Eat (Cooldown)
class Predator(Organism):
    num_predator=0
    def __init__(self, position, neuralnet):
        super().__init__(position, neuralnet)
        Predator.num_predator+=1
        if Predator.num_predator>max_predator:
            self.death()
        self.sense_range=predator_sense_range
        self.sense_angle=predator_sense_angle
        self.color=(255,0,0)
        self.eat_counter=0
        self.eat_on_cooldown=True
        self.eat_cooldown_timer=0

    def is_prey(self):
        return False
    def eat(self):
        for organism in organisms:
            distance_to_organism = ((organism.position[0] - self.position[0]) ** 2 + (-organism.position[1] +self.position[1]) ** 2) ** 0.5
            if 2*organism_radius>=distance_to_organism and distance_to_organism!=0:
                if organism.is_prey() and self.eat_on_cooldown==False:
                    self.eat_counter+=1
                    self.energy+=predator_energy_gain_on_eat
                    self.eat_on_cooldown=True
                    organism.death()
                    return
    def iterate(self):
        super().iterate()
        if self.eat_on_cooldown==True:
            if self.eat_cooldown_timer<predator_eat_cooldown:
                self.eat_cooldown_timer+=1/frame_rate
            else:
                self.eat_cooldown_timer=0
                self.eat_on_cooldown=False
        if self.eat_counter>=predator_split_eat_amount:
            self.eat_counter=0
            self.split()
        self.eat()
        if self.energy<=0:
            self.death()
        elif self.energy >=max_energy:
            self.energy=max_energy
        self.energy-=predator_energy_degen
    def split(self):
        super().split()
        new_neuralnet = self.neuralnet.mutate()
        organisms.append(Predator(self.position+np.random.uniform(-2*organism_radius, 2*organism_radius, size=2), new_neuralnet))
    def death(self):
        super().death()
        Predator.num_predator-=1
        if (Predator.num_predator==0):
            organisms.append(Predator([random.random()*window_size[0],random.random()*window_size[1]], self.neuralnet))

import random
import copy


class NeuralNetwork:
    def __init__(self, input, hidden, output, hidden_layers):
        # Constructor method
        self.input_nodes=input
        self.hidden_nodes=hidden
        self.output_nodes=output
        self.hidden_layers=hidden_layers

        self.weights = []
        self.weights.append(np.random.uniform(-1, 1, size=(self.hidden_nodes, self.input_nodes + 1)))
        for i in range(1, self.hidden_layers + 1):
            self.weights.append(np.random.uniform(-1, 1, size=(self.hidden_nodes, self.hidden_nodes + 1)))
        self.weights.append(np.random.uniform(-1, 1, size=(self.output_nodes, self.hidden_nodes + 1)))



    def mutate(self, mutation_rate=mutation_rate):
        clone=self.clone()
        for n in range (len(clone.weights)):
            for i in range(clone.weights[n].shape[0]):
                for j in range(clone.weights[n].shape[1]):
                    if(random.random()<mutation_rate):
                        clone.weights[n][i][j]+=np.random.normal(0,0.2)
                        if (clone.weights[n][i][j]>1):
                            clone.weights[n][i][j]=1
                        elif (clone.weights[n][i][j]<-1):
                            clone.weights[n][i][j]=-1
        return clone

    def calculate(self, inputs):
        output = np.append(inputs, 1)
        for i in range(self.hidden_layers + 2):
            output = np.maximum(np.dot(self.weights[i], output), 0)
            if i != self.hidden_layers + 1: 
                output = np.append(output, 1)

        return output

    def clone(self):
        clone = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes, self.hidden_layers)
        clone.weights = copy.deepcopy(self.weights)
        return clone
    




pygame.init()

# Set up the Pygame display
window = pygame.display.set_mode(window_size)
pygame.display.set_caption("Neural Net")
# Create a GUI manager
gui_manager = pygame_gui.UIManager(window_size)
# Game loop
running = True
clock = pygame.time.Clock()
frames=0



organisms=[]
for i in range(num_organisms_start):
    neural_net = NeuralNetwork(num_sensor_lines, 16, 2, 2)
    if random.random()<=max_prey/(max_prey+max_predator):
        organisms.append(Prey([random.random()*window_size[0],random.random()*window_size[1]], neural_net))
    else:
        organisms.append(Predator([random.random()*window_size[0],random.random()*window_size[1]], neural_net))


while running:
    time_delta = clock.tick(frame_rate) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        gui_manager.process_events(event)
    for organism in organisms:
        organism.iterate()

    organisms = [organism for organism in organisms if organism.is_alive]

    print(Predator.num_predator)
    print(Prey.num_prey)
    print("#####")
    frames+=1
    gui_manager.update(time_delta)
    window.fill((0, 0, 0))
    for organism in organisms:
        organism.draw()
    window.blit(pygame.font.Font(None, 20).render(str(frames), True, (255, 255, 255)), (10, 10))
    gui_manager.draw_ui(window)
    pygame.display.update()

pygame.quit()




