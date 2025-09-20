import traci
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Path to your SUMO executable
SUMO_BINARY = "sumo-gui"  # or full path

# Start SUMO with your network config
traci.start([SUMO_BINARY, "-c", "my_network.sumocfg"])

fig, ax = plt.subplots()
scat = ax.scatter([], [])

def update(frame):
    traci.simulationStep()
    vehicle_ids = traci.vehicle.getIDList()
    x = [traci.vehicle.getPosition(vid)[0] for vid in vehicle_ids]
    y = [traci.vehicle.getPosition(vid)[1] for vid in vehicle_ids]
    
    scat.set_offsets(list(zip(x, y)))
    ax.set_xlim(0, 100)  # adjust to your network size
    ax.set_ylim(0, 100)
    ax.set_title(f"Step {frame}, Vehicles: {len(vehicle_ids)}")
    return scat,

ani = animation.FuncAnimation(fig, update, frames=100, interval=200)
plt.show()

traci.close()
