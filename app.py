import json
import re
import requests
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

MODEL = "gemma4:e2b"
OLLAMA_URL = "http://localhost:11434/api/chat"

G = 6.67430e-11
PLANET_RADIUS = 6.4e6
START_RADIUS = 1.0e7

st.set_page_config(
    page_title="AI Physics Challenge",
    layout="wide",
)

st.title("AI Physics Challenge")
st.caption("AI sets the mission. You control the physics.")

st.write(
    "Your mission is to manipulate a satellite simulation until "
    "you achieve the outcome requested by the AI. Change the "
    "planet's mass, adjust the satellite's speed, run the "
    "simulation, and see whether you completed the challenge."
)

# Make a new mission with the AI

def generate_challenge():
    prompt = """
Create one interesting, achievable high-school physics challenge
about a satellite orbiting a planet.

The student can change:
- Planet mass: 0.5 to 8.0 times 10^24 kg
- Satellite starting speed: 1000 to 12000 m/s
- Simulation duration: 1 to 6 hours

The satellite starts 10,000 km from the planet's center.
The planet radius is 6400 km.
The planet remains fixed at the origin.
The simulation uses Newtonian gravity.

Choose exactly one target_outcome:
- "impact": the satellite hits the planet during the simulation.
- "bound": the satellite avoids impact and has negative specific
  mechanical energy at the end of the simulation.
- "unbound": the satellite avoids impact and has nonnegative
  specific mechanical energy at the end of the simulation.

The challenge must be achievable using the controls.
Choose a target_duration_hours from 1 through 6.

Return ONLY valid JSON in this exact format:
{
  "title": "Short mission title",
  "story": "An engaging scenario in 2-3 sentences.",
  "objective": "A clear description of the required outcome.",
  "target_outcome": "bound",
  "target_duration_hours": 3
}

Replace the example outcome with your actual choice.
Do not include Markdown fences or any text outside the JSON.
Keep the mission scientifically consistent with the rules above.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "format": "json",
        },
        timeout=300,
    )
    response.raise_for_status()

    content = response.json()["message"]["content"].strip()

    # Sometimes the model adds code fences around the JSON, so strip those first.
    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content,
        flags=re.IGNORECASE,
    ).strip()

    challenge = json.loads(content)

    allowed_outcomes = {"impact", "bound", "unbound"}

    if challenge.get("target_outcome") not in allowed_outcomes:
        raise ValueError("AI returned an invalid target outcome.")

    duration = int(challenge.get("target_duration_hours", 0))
    if not 1 <= duration <= 6:
        raise ValueError("AI returned an invalid mission duration.")

    for field in ("title", "story", "objective"):
        if not isinstance(challenge.get(field), str):
            raise ValueError(f"AI did not provide a valid {field}.")

    return challenge


# Satellite physics

def acceleration(position, mass):
    distance = np.linalg.norm(position)

    if distance == 0:
        return np.zeros(2)

    return -G * mass * position / distance**3


def run_simulation(mass, speed, duration_hours):
    dt = 5.0
    steps = int(duration_hours * 3600 / dt)

    position = np.array([START_RADIUS, 0.0])
    velocity = np.array([0.0, float(speed)])

    positions = [position.copy()]
    speeds = [np.linalg.norm(velocity)]
    distances = [np.linalg.norm(position)]

    collision = False

    for _ in range(steps):
        a1 = acceleration(position, mass)

        next_position = (
            position + velocity * dt + 0.5 * a1 * dt**2
        )

        if np.linalg.norm(next_position) <= PLANET_RADIUS:
            positions.append(next_position.copy())
            speeds.append(np.linalg.norm(velocity))
            distances.append(np.linalg.norm(next_position))
            collision = True
            break

        a2 = acceleration(next_position, mass)

        next_velocity = velocity + 0.5 * (a1 + a2) * dt

        position = next_position
        velocity = next_velocity

        positions.append(position.copy())
        speeds.append(np.linalg.norm(velocity))
        distances.append(np.linalg.norm(position))

    positions = np.array(positions)
    speeds = np.array(speeds)
    distances = np.array(distances)

    energy = (
        0.5 * speeds[-1] ** 2
        - G * mass / distances[-1]
    )

    if collision:
        outcome = "impact"
    elif energy < 0:
        outcome = "bound"
    else:
        outcome = "unbound"

    # Unwrap the angles so they don't jump when the satellite crosses the -pi/pi line.
    angles = np.unwrap(
        np.arctan2(positions[:, 1], positions[:, 0])
    )
    revolutions = abs(angles[-1] - angles[0]) / (2 * np.pi)

    return {
        "positions": positions,
        "distances": distances,
        "speeds": speeds,
        "energy": energy,
        "collision": collision,
        "outcome": outcome,
        "revolutions": revolutions,
        "elapsed_hours": (len(positions) - 1) * dt / 3600,
    }


# Mission setup

st.divider()
st.header("Mission Control")

if st.button("Generate a new AI mission", type="primary"):
    with st.spinner("AI is designing your next mission..."):
        try:
            st.session_state["challenge"] = generate_challenge()
            st.session_state.pop("last_result", None)
            st.session_state.pop("last_settings", None)
            st.session_state.pop("feedback", None)

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to Ollama. Make sure Ollama "
                "is running and your model is installed."
            )
        except requests.exceptions.Timeout:
            st.error("The AI took too long to generate a mission.")
        except (
            requests.exceptions.RequestException,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as error:
            st.error(f"Could not generate a valid mission: {error}")

challenge = st.session_state.get("challenge")

if challenge is None:
    st.info(
        "Generate your first mission to begin. Your objective "
        "will appear here."
    )
    st.stop()

st.subheader(challenge["title"])
st.write(challenge["story"])

st.markdown("**YOUR OBJECTIVE**")
st.write(challenge["objective"])

st.markdown(
    f"**Required simulation duration:** "
    f"{challenge['target_duration_hours']} hour(s)"
)

st.caption(
    "Mission rules: impact means a collision; bound means "
    "negative specific energy without a collision; unbound means "
    "nonnegative specific energy without a collision."
)

# Change the settings for your attempt

st.divider()
st.header("Configure Your Satellite")

left, right = st.columns([1, 1.4])

with left:
    st.write("### Experiment controls")

    mass_setting = st.slider(
        "Planet mass (10²⁴ kg)",
        min_value=0.5,
        max_value=8.0,
        value=4.0,
        step=0.5,
        key="planet_mass",
    )

    speed_setting = st.slider(
        "Initial satellite speed (m/s)",
        min_value=1000,
        max_value=12000,
        value=5000,
        step=250,
        key="satellite_speed",
    )

    duration_setting = st.slider(
        "Simulation duration (hours)",
        min_value=1,
        max_value=6,
        value=3,
        step=1,
        key="simulation_duration",
    )

    st.caption(
        "The initial velocity is perpendicular to the satellite's "
        "starting position. The starting distance remains fixed."
    )

    run_clicked = st.button(
        "Run simulation",
        type="primary",
        use_container_width=True,
    )

with right:
    st.write("### Current settings")

    st.metric(
        "Planet mass",
        f"{mass_setting:.1f} × 10²⁴ kg",
    )
    st.metric(
        "Initial speed",
        f"{speed_setting:,} m/s",
    )
    st.metric(
        "Simulation duration",
        f"{duration_setting} hours",
    )

    if duration_setting < challenge["target_duration_hours"]:
        st.warning(
            "Your simulation is shorter than the mission requires. "
            "Increase its duration before attempting the challenge."
        )

# Run the simulation and check the mission requirements

if run_clicked:
    result = run_simulation(
        mass_setting * 1e24,
        speed_setting,
        duration_setting,
    )

    settings = {
        "mass": mass_setting,
        "speed": speed_setting,
        "duration": duration_setting,
    }

    passed_duration = (
        result["elapsed_hours"]
        >= challenge["target_duration_hours"] - 0.01
    )

    passed_outcome = (
        result["outcome"] == challenge["target_outcome"]
    )

    success = passed_duration and passed_outcome

    st.session_state["last_result"] = result
    st.session_state["last_settings"] = settings
    st.session_state["feedback"] = {
        "success": success,
        "passed_duration": passed_duration,
        "passed_outcome": passed_outcome,
    }

# Show what happened

result = st.session_state.get("last_result")

if result is not None:
    feedback = st.session_state["feedback"]

    st.divider()
    st.header("Mission Results")

    if feedback["success"]:
        st.success(
            "MISSION COMPLETE! You achieved the required outcome "
            "for the required duration."
        )
    else:
        st.error("Mission incomplete. Adjust your controls and try again.")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Actual outcome",
        result["outcome"].upper(),
    )

    col2.metric(
        "Completed duration",
        f"{result['elapsed_hours']:.2f} h",
    )

    col3.metric(
        "Revolutions around planet",
        f"{result['revolutions']:.2f}",
    )

    st.write("### Mission checklist")

    if feedback["passed_outcome"]:
        st.write("PASS: The simulated outcome matches the mission.")
    else:
        st.write(
            f"NOT YET: The mission requires "
            f"{challenge['target_outcome']}, but the simulation "
            f"produced {result['outcome']}."
        )

    if feedback["passed_duration"]:
        st.write("PASS: The simulation ran for the required duration.")
    else:
        st.write(
            "NOT YET: The simulation did not last long enough "
            "to meet the mission requirement."
        )

    # Plot the path the satellite took
    positions = result["positions"]

    fig, ax = plt.subplots(figsize=(8, 7))

    planet = plt.Circle(
        (0, 0),
        PLANET_RADIUS,
        color="steelblue",
        alpha=0.8,
        label="Planet",
    )
    ax.add_patch(planet)

    ax.plot(
        positions[:, 0],
        positions[:, 1],
        label="Satellite trajectory",
    )

    ax.scatter(
        positions[0, 0],
        positions[0, 1],
        color="green",
        label="Start",
    )

    ax.scatter(
        positions[-1, 0],
        positions[-1, 1],
        color="red",
        marker="x",
        label="Finish",
    )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Horizontal position (m)")
    ax.set_ylabel("Vertical position (m)")
    ax.set_title("Your Experimental Trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Inspect numerical evidence"):
        st.write(
            f"Final distance from planet center: "
            f"{result['distances'][-1]:,.0f} m"
        )
        st.write(
            f"Closest recorded distance to planet center: "
            f"{np.min(result['distances']):,.0f} m"
        )
        st.write(
            f"Final specific mechanical energy: "
            f"{result['energy']:.4e} J/kg"
        )
        st.write(
            f"Collision detected: {result['collision']}"
        )

    # Ask the AI to explain the results

    st.divider()
    st.subheader("AI Mission Coach")

    if st.button("Explain my results with AI"):
        prompt = f"""
You are a high-school physics mission coach.

The student attempted this challenge:
{json.dumps(challenge)}

Their settings:
{json.dumps(st.session_state['last_settings'])}

Computed simulation results:
Outcome: {result['outcome']}
Elapsed simulation time: {result['elapsed_hours']:.3f} hours
Revolutions: {result['revolutions']:.3f}
Final specific energy: {result['energy']:.6e} J/kg
Collision detected: {result['collision']}
Mission completed: {feedback['success']}

Explain:
1. Why the satellite produced this outcome.
2. Whether the mission was completed and which conditions passed.
3. One specific adjustment the student could try next.
4. The relevant physics behind that adjustment.

Use the supplied results as the source of truth.
Do not invent measurements or claim the student succeeded
if the computed mission check failed.
Explain that the simulation uses a fixed planet and idealized
Newtonian gravity. Be concise, specific, and encouraging.
"""

        with st.spinner("AI is analyzing your attempt..."):
            try:
                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": MODEL,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "stream": False,
                    },
                    timeout=300,
                )
                response.raise_for_status()

                st.markdown(
                    response.json()["message"]["content"]
                )

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to Ollama.")
            except requests.exceptions.Timeout:
                st.error("The AI coach took too long to respond.")
            except (
                requests.exceptions.RequestException,
                KeyError,
                ValueError,
            ) as error:
                st.error(f"Could not retrieve AI feedback: {error}")

