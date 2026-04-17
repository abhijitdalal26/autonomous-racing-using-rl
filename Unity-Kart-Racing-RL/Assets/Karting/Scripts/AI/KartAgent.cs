using KartGame.KartSystems;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using Random = UnityEngine.Random;

namespace KartGame.AI
{
    /// <summary>
    /// Sensors hold information such as the position of rotation of the origin of the raycast and its hit threshold
    /// to consider a "crash".
    /// </summary>
    [System.Serializable]
    public struct Sensor
    {
        public Transform Transform;
        public float RayDistance;
        public float HitValidationDistance;
    }

    /// <summary>
    /// We only want certain behaviours when the agent runs.
    /// Training would allow certain functions such as OnAgentReset() be called and execute, while Inferencing will
    /// assume that the agent will continuously run and not reset.
    /// </summary>
    public enum AgentMode
    {
        Training,
        Inferencing
    }

    /// <summary>
    /// The KartAgent will drive the inputs for the KartController.
    /// </summary>
    public class KartAgent : Agent, IInput
    {
        const int k_DefaultRespawnMask = (1 << 9) | (1 << 10) | (1 << 11);

#region Training Modes
        [Tooltip("Are we training the agent or is the agent production ready?")]
        public AgentMode Mode = AgentMode.Training;
        [Tooltip("What is the initial checkpoint the agent will go to? This value is only for inferencing.")]
        public ushort InitCheckpointIndex;

#endregion

#region Senses
        [Header("Observation Params")]
        [Tooltip("What objects should the raycasts hit and detect?")]
        public LayerMask Mask;
        [Tooltip("Sensors contain ray information to sense out the world, you can have as many sensors as you need.")]
        public Sensor[] Sensors;
        [Header("Checkpoints"), Tooltip("What are the series of checkpoints for the agent to seek and pass through?")]
        public Collider[] Colliders;
        [Tooltip("What layer are the checkpoints on? This should be an exclusive layer for the agent to use.")]
        public LayerMask CheckpointMask;

        [Space]
        [Tooltip("Would the agent need a custom transform to be able to raycast and hit the track? " +
            "If not assigned, then the root transform will be used.")]
        public Transform AgentSensorTransform;
#endregion

#region Rewards
        [Header("Rewards"), Tooltip("What penatly is given when the agent crashes?")]
        public float HitPenalty = -1f;
        [Tooltip("How much reward is given when the agent successfully passes the checkpoints?")]
        public float PassCheckpointReward;
        [Tooltip("Should typically be a small value, but we reward the agent for moving in the right direction.")]
        public float TowardsCheckpointReward;
        [Tooltip("Typically if the agent moves faster, we want to reward it for finishing the track quickly.")]
        public float SpeedReward;
        [Tooltip("Reward the agent when it keeps accelerating")]
        public float AccelerationReward;
        #endregion

        #region ResetParams
        [Header("Inference Reset Params")]
        [Tooltip("What is the unique mask that the agent should detect when it falls out of the track?")]
        public LayerMask OutOfBoundsMask;
        [Tooltip("What are the layers we want to detect for the track and the ground?")]
        public LayerMask TrackMask;
        [Tooltip("How far should the ray be when casted? For larger karts - this value should be larger too.")]
        public float GroundCastDistance;
        [Tooltip("How high above the detected ground to place the kart when resetting during training.")]
        public float RespawnHeight = 0.5f;
        [Tooltip("How far above a checkpoint to search for valid track ground during training resets.")]
        public float RespawnProbeHeight = 10f;
        [Tooltip("How far downward to search for the track when resetting during training.")]
        public float RespawnProbeDistance = 30f;
#endregion

#region Debugging
        [Header("Debug Option")] [Tooltip("Should we visualize the rays that the agent draws?")]
        public bool ShowRaycasts;
#endregion

        ArcadeKart m_Kart;
        bool m_Acceleration;
        bool m_Brake;
        float m_Steering;
        int m_CheckpointIndex;

        bool m_EndEpisode;
        float m_LastAccumulatedReward;

        void Awake()
        {
            m_Kart = GetComponent<ArcadeKart>();
            if (AgentSensorTransform == null) AgentSensorTransform = transform;
        }

        void Start()
        {
            // If the agent is training, then at the start of the simulation, pick a random checkpoint to train the agent.
            OnEpisodeBegin();

            if (Mode == AgentMode.Inferencing) m_CheckpointIndex = InitCheckpointIndex;
        }

        void Update()
        {
            if (m_EndEpisode)
            {
                m_EndEpisode = false;
                AddReward(m_LastAccumulatedReward);
                EndEpisode();
                OnEpisodeBegin();
            }
        }

        void LateUpdate()
        {
            switch (Mode)
            {
                case AgentMode.Inferencing:
                    if (ShowRaycasts) 
                        Debug.DrawRay(transform.position, Vector3.down * GroundCastDistance, Color.cyan);

                    // We want to place the agent back on the track if the agent happens to launch itself outside of the track.
                    if (Physics.Raycast(transform.position + Vector3.up, Vector3.down, out var hit, GroundCastDistance, TrackMask)
                        && ((1 << hit.collider.gameObject.layer) & OutOfBoundsMask) > 0)
                    {
                        // Reset the agent back to its last known checkpoint if it is still valid.
                        if (TryGetCheckpointCollider(m_CheckpointIndex, out var checkpointCollider))
                        {
                            ResetToCheckpoint(checkpointCollider);
                            m_Kart.Rigidbody.linearVelocity = default;
                            m_Kart.Rigidbody.angularVelocity = default;
                        }

                        m_Steering = 0f;
						m_Acceleration = m_Brake = false; 
                    }

                    break;
            }
        }

        void OnTriggerEnter(Collider other)
        {
            if (!EnsureValidCheckpoints() || other == null)
                return;

            var maskedValue = 1 << other.gameObject.layer;
            var triggered = maskedValue & CheckpointMask;

            FindCheckpointIndex(other, out var index);

            // Ensure that the agent touched the checkpoint and the new index is greater than the m_CheckpointIndex.
            if (triggered > 0 && index > m_CheckpointIndex || index == 0 && m_CheckpointIndex == Colliders.Length - 1)
            {
                AddReward(PassCheckpointReward);
                m_CheckpointIndex = index;
            }
        }

        void FindCheckpointIndex(Collider checkPoint, out int index)
        {
            if (!EnsureValidCheckpoints() || checkPoint == null)
            {
                index = -1;
                return;
            }

            for (int i = 0; i < Colliders.Length; i++)
            {
                if (Colliders[i] != null && Colliders[i].GetInstanceID() == checkPoint.GetInstanceID())
                {
                    index = i;
                    return;
                }
            }
            index = -1;
        }

        float Sign(float value)
        {
            if (value > 0)
            {
                return 1;
            } 
            if (value < 0)
            {
                return -1;
            }
            return 0;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            sensor.AddObservation(m_Kart.LocalSpeed());

            // Add an observation for direction of the agent to the next checkpoint.
            if (EnsureValidCheckpoints())
            {
                var next = (m_CheckpointIndex + 1) % Colliders.Length;
                var nextCollider = Colliders[next];
                if (nextCollider != null)
                {
                    var direction = (nextCollider.transform.position - m_Kart.transform.position).normalized;
                    sensor.AddObservation(Vector3.Dot(m_Kart.Rigidbody.linearVelocity.normalized, direction));

                    if (ShowRaycasts)
                        Debug.DrawLine(AgentSensorTransform.position, nextCollider.transform.position, Color.magenta);
                }
            }
            else
            {
                sensor.AddObservation(0f);
            }

            m_LastAccumulatedReward = 0.0f;
            m_EndEpisode = false;
            for (var i = 0; i < Sensors.Length; i++)
            {
                var current = Sensors[i];
                var xform = current.Transform;
                var hit = Physics.Raycast(AgentSensorTransform.position, xform.forward, out var hitInfo,
                    current.RayDistance, Mask, QueryTriggerInteraction.Ignore);

                if (ShowRaycasts)
                {
                    Debug.DrawRay(AgentSensorTransform.position, xform.forward * current.RayDistance, Color.green);
                    Debug.DrawRay(AgentSensorTransform.position, xform.forward * current.HitValidationDistance, 
                        Color.red);

                    if (hit && hitInfo.distance < current.HitValidationDistance)
                    {
                        Debug.DrawRay(hitInfo.point, Vector3.up * 3.0f, Color.blue);
                    }
                }

                if (hit)
                {
                    if (hitInfo.distance < current.HitValidationDistance)
                    {
                        m_LastAccumulatedReward += HitPenalty;
                        m_EndEpisode = true;
                    }
                }

                sensor.AddObservation(hit ? hitInfo.distance : current.RayDistance);
            }

            sensor.AddObservation(m_Acceleration);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            base.OnActionReceived(actions);
            InterpretDiscreteActions(actions);

            // Find the next checkpoint when registering the current checkpoint that the agent has passed.
            if (EnsureValidCheckpoints())
            {
                var next = (m_CheckpointIndex + 1) % Colliders.Length;
                var nextCollider = Colliders[next];
                if (nextCollider != null)
                {
                    var direction = (nextCollider.transform.position - m_Kart.transform.position).normalized;
                    var reward = Vector3.Dot(m_Kart.Rigidbody.linearVelocity.normalized, direction);

                    if (ShowRaycasts) Debug.DrawRay(AgentSensorTransform.position, m_Kart.Rigidbody.linearVelocity, Color.blue);

                    // Add rewards if the agent is heading in the right direction
                    AddReward(reward * TowardsCheckpointReward);
                }
            }

            AddReward((m_Acceleration && !m_Brake ? 1.0f : 0.0f) * AccelerationReward);
            AddReward(m_Kart.LocalSpeed() * SpeedReward);
        }

        public override void OnEpisodeBegin()
        {
            switch (Mode)
            {
                case AgentMode.Training:
                    if (!EnsureValidCheckpoints())
                    {
                        Debug.LogWarning("No colliders (checkpoints) assigned to KartAgent! Please assign them in the Inspector.");
                        return;
                    }
                    m_CheckpointIndex = Random.Range(0, Colliders.Length);
                    var collider = Colliders[m_CheckpointIndex];
                    ResetToCheckpoint(collider);
                    m_Kart.Rigidbody.linearVelocity = default;
                    m_Kart.Rigidbody.angularVelocity = default;
                    m_Acceleration = false;
                    m_Brake = false;
                    m_Steering = 0f;
                    break;
                default:
                    break;
            }
        }

        void InterpretDiscreteActions(ActionBuffers actions)
        {
            m_Steering = actions.DiscreteActions[0] - 1f;
            m_Acceleration = actions.DiscreteActions[1] >= 1.0f;
            m_Brake = actions.DiscreteActions[1] < 1.0f;
        }

        public InputData GenerateInput()
        {
            return new InputData
            {
                Accelerate = m_Acceleration,
                Brake = m_Brake,
                TurnInput = m_Steering
            };
        }

        void ResetToCheckpoint(Collider checkpointCollider)
        {
            var checkpointTransform = checkpointCollider.transform;
            var spawnRotation = Quaternion.Euler(0f, checkpointTransform.eulerAngles.y, 0f);
            var spawnPosition = checkpointCollider.bounds.center + Vector3.up * (checkpointCollider.bounds.extents.y + RespawnHeight);

            var rayOrigin = checkpointCollider.bounds.center + Vector3.up * RespawnProbeHeight;
            var respawnMask = TrackMask.value != 0 ? TrackMask.value : k_DefaultRespawnMask;
            if (Physics.Raycast(rayOrigin, Vector3.down, out var hit, RespawnProbeDistance, respawnMask, QueryTriggerInteraction.Ignore))
            {
                spawnPosition = hit.point + Vector3.up * RespawnHeight;
                var projectedForward = Vector3.ProjectOnPlane(checkpointTransform.forward, hit.normal).normalized;
                if (projectedForward.sqrMagnitude > 0.0001f)
                {
                    spawnRotation = Quaternion.LookRotation(projectedForward, hit.normal);
                }
            }

            transform.SetPositionAndRotation(spawnPosition, spawnRotation);
        }

        bool EnsureValidCheckpoints()
        {
            if (Colliders == null || Colliders.Length == 0)
                return false;

            var validColliders = new List<Collider>(Colliders.Length);
            for (int i = 0; i < Colliders.Length; i++)
            {
                if (Colliders[i] != null)
                    validColliders.Add(Colliders[i]);
            }

            if (validColliders.Count == 0)
            {
                Colliders = null;
                m_CheckpointIndex = 0;
                return false;
            }

            if (validColliders.Count != Colliders.Length)
            {
                Colliders = validColliders.ToArray();
            }

            if (m_CheckpointIndex >= Colliders.Length)
            {
                m_CheckpointIndex = 0;
            }

            return true;
        }

        bool TryGetCheckpointCollider(int index, out Collider checkpointCollider)
        {
            checkpointCollider = null;
            if (!EnsureValidCheckpoints())
                return false;

            if (index < 0 || index >= Colliders.Length)
                return false;

            checkpointCollider = Colliders[index];
            return checkpointCollider != null;
        }
    }
}
