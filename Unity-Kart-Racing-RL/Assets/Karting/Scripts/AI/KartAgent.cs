using KartGame.KartSystems;
using System;
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
        [Tooltip("When enabled, training episodes can begin from any checkpoint. Disable for full-lap training from the start line.")]
        public bool RandomizeTrainingStartCheckpoint;
        [Tooltip("Checkpoint index used for training when random starts are disabled. Set to 0 for the starting line.")]
        public ushort TrainingStartCheckpointIndex;
        [Tooltip("When enabled, agents that were duplicated from prefabs can rebuild their checkpoint list from the scene.")]
        public bool AutoAssignSceneCheckpoints = true;
        [Tooltip("How far apart training karts should be placed sideways when they share the same checkpoint reset.")]
        public float TrainingSpawnSpacing = 4f;
        [Tooltip("How far behind the start line each extra training row should be placed.")]
        public float TrainingSpawnRowSpacing = 6f;
        [Tooltip("Maximum number of training karts to place side by side before starting a new row.")]
        public int TrainingSpawnColumns = 3;
        [Tooltip("How many training action steps to ignore crash-ending ray hits after each reset.")]
        public int TrainingResetGraceSteps = 25;
        [Tooltip("When enabled, a training episode ends as soon as the agent reaches the last checkpoint in the ordered list.")]
        public bool EndEpisodeOnLastCheckpoint = true;
        [Tooltip("Optional explicit training track config. If empty, the scene start line and checkpoints are auto-discovered.")]
        public TrainingTrackConfig TrackConfig;

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
        [Tooltip("Penalty applied when the agent touches a checkpoint out of order, such as going backward.")]
        public float WrongCheckpointPenalty = -0.5f;
        [Tooltip("When enabled, touching a checkpoint out of order immediately ends the current training episode.")]
        public bool EndEpisodeOnWrongCheckpoint = true;
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
        int m_StepsSinceReset;

        void Awake()
        {
            m_Kart = GetComponent<ArcadeKart>();
            if (AgentSensorTransform == null) AgentSensorTransform = transform;
            if (TrackConfig == null) TrackConfig = TrainingTrackConfig.Resolve();
            RefreshTrainingCheckpointsFromScene();
        }

        void Start()
        {
            if (Mode == AgentMode.Inferencing)
            {
                if (!EnsureValidCheckpoints())
                {
                    Debug.LogWarning("No colliders (checkpoints) assigned to KartAgent! Please assign them in the Inspector.");
                    return;
                }

                m_CheckpointIndex = ClampCheckpointIndex(InitCheckpointIndex);
                if (m_CheckpointIndex == 0 && TryResetToStartLine())
                {
                    ClearMotionAndInput();
                }
                else if (TryGetCheckpointCollider(m_CheckpointIndex, out var checkpointCollider))
                {
                    ResetToCheckpoint(checkpointCollider);
                    ClearMotionAndInput();
                }

                return;
            }

            OnEpisodeBegin();
        }

        void Update()
        {
            if (m_EndEpisode)
            {
                m_EndEpisode = false;
                AddReward(m_LastAccumulatedReward);
                EndEpisode();
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

            if (triggered <= 0 || index < 0 || Colliders == null || Colliders.Length == 0)
                return;

            var expectedCheckpointIndex = (m_CheckpointIndex + 1) % Colliders.Length;
            var touchedExpectedCheckpoint = index == expectedCheckpointIndex;
            var touchedLastCheckpoint = index == Colliders.Length - 1;

            if (touchedExpectedCheckpoint)
            {
                AddReward(PassCheckpointReward);
                m_CheckpointIndex = index;

                if (Mode == AgentMode.Training && EndEpisodeOnLastCheckpoint && touchedLastCheckpoint)
                {
                    EndEpisode();
                }

                return;
            }

            if (Mode == AgentMode.Training)
            {
                AddReward(WrongCheckpointPenalty);
                if (EndEpisodeOnWrongCheckpoint)
                {
                    EndEpisode();
                }
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
                        if (!IsWithinTrainingResetGracePeriod())
                        {
                            m_LastAccumulatedReward += HitPenalty;
                            m_EndEpisode = true;
                        }
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
            m_StepsSinceReset++;
        }

        public override void OnEpisodeBegin()
        {
            switch (Mode)
            {
                case AgentMode.Training:
                    RefreshTrainingCheckpointsFromScene();
                    if (!EnsureValidCheckpoints())
                    {
                        Debug.LogWarning("No colliders (checkpoints) assigned to KartAgent! Please assign them in the Inspector.");
                        return;
                    }
                    m_CheckpointIndex = GetTrainingStartCheckpointIndex();
                    if (!RandomizeTrainingStartCheckpoint && m_CheckpointIndex == 0 && TryResetToStartLine())
                    {
                        ClearMotionAndInput();
                        break;
                    }

                    var collider = Colliders[m_CheckpointIndex];
                    ResetToCheckpoint(collider);
                    ClearMotionAndInput();
                    break;
                default:
                    break;
            }
        }

        void InterpretDiscreteActions(ActionBuffers actions)
        {
            m_Steering = actions.DiscreteActions[0] - 1f;
            var throttleAction = actions.DiscreteActions[1];
            m_Brake = throttleAction == 0;
            m_Acceleration = throttleAction == 2;
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

            spawnPosition += GetTrainingSpawnOffset(spawnRotation);
            transform.SetPositionAndRotation(spawnPosition, spawnRotation);
        }

        bool EnsureValidCheckpoints()
        {
            if ((Colliders == null || Colliders.Length == 0) && !TryAssignSceneCheckpoints())
                return false;

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
                return TryAssignSceneCheckpoints();
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

        void RefreshTrainingCheckpointsFromScene()
        {
            if (Mode != AgentMode.Training || !AutoAssignSceneCheckpoints)
                return;

            var activeTrackConfig = ResolveTrackConfig();
            if (activeTrackConfig != null && activeTrackConfig.TryGetOrderedCheckpoints(out var discoveredCheckpoints))
            {
                Colliders = discoveredCheckpoints;
                m_CheckpointIndex = ClampCheckpointIndex(m_CheckpointIndex);
            }
        }

        bool TryAssignSceneCheckpoints()
        {
            if (!(AutoAssignSceneCheckpoints || Mode == AgentMode.Training))
                return false;

            var activeTrackConfig = ResolveTrackConfig();
            if (activeTrackConfig == null || !activeTrackConfig.TryGetOrderedCheckpoints(out var discoveredCheckpoints))
                return false;

            Colliders = discoveredCheckpoints;
            m_CheckpointIndex = ClampCheckpointIndex(m_CheckpointIndex);
            return true;
        }

        TrainingTrackConfig ResolveTrackConfig()
        {
            if (TrackConfig == null)
                TrackConfig = TrainingTrackConfig.Resolve();

            return TrackConfig;
        }

        bool TryResetToStartLine()
        {
            var activeTrackConfig = ResolveTrackConfig();
            if (activeTrackConfig == null || !activeTrackConfig.TryGetStartLine(out var startLineTransform, out var startLineCollider))
                return false;

            var spawnRotation = Quaternion.Euler(0f, startLineTransform.eulerAngles.y, 0f);
            var spawnAnchor = startLineCollider != null ? startLineCollider.bounds.center : startLineTransform.position;
            var spawnPosition = spawnAnchor + Vector3.up * RespawnHeight;

            var rayOrigin = spawnAnchor + Vector3.up * RespawnProbeHeight;
            var respawnMask = TrackMask.value != 0 ? TrackMask.value : k_DefaultRespawnMask;
            if (Physics.Raycast(rayOrigin, Vector3.down, out var hit, RespawnProbeDistance, respawnMask, QueryTriggerInteraction.Ignore))
            {
                spawnPosition = hit.point + Vector3.up * RespawnHeight;
                var projectedForward = Vector3.ProjectOnPlane(startLineTransform.forward, hit.normal).normalized;
                if (projectedForward.sqrMagnitude > 0.0001f)
                {
                    spawnRotation = Quaternion.LookRotation(projectedForward, hit.normal);
                }
            }

            spawnPosition -= spawnRotation * Vector3.forward * Mathf.Max(0f, activeTrackConfig.StartLineBackOffset);
            spawnPosition += GetTrainingSpawnOffset(spawnRotation);
            transform.SetPositionAndRotation(spawnPosition, spawnRotation);
            return true;
        }

        int GetTrainingStartCheckpointIndex()
        {
            if (Colliders == null || Colliders.Length == 0)
                return 0;

            if (RandomizeTrainingStartCheckpoint)
                return Random.Range(0, Colliders.Length);

            return ClampCheckpointIndex(TrainingStartCheckpointIndex);
        }

        int ClampCheckpointIndex(int index)
        {
            if (Colliders == null || Colliders.Length == 0)
                return 0;

            return Mathf.Clamp(index, 0, Colliders.Length - 1);
        }

        void ClearMotionAndInput()
        {
            m_Kart.Rigidbody.linearVelocity = default;
            m_Kart.Rigidbody.angularVelocity = default;
            m_Acceleration = false;
            m_Brake = false;
            m_Steering = 0f;
            m_StepsSinceReset = 0;
        }

        bool IsWithinTrainingResetGracePeriod()
        {
            return Mode == AgentMode.Training && m_StepsSinceReset < Mathf.Max(0, TrainingResetGraceSteps);
        }

        Vector3 GetTrainingSpawnOffset(Quaternion spawnRotation)
        {
            if (Mode != AgentMode.Training || TrainingSpawnSpacing <= 0f)
                return Vector3.zero;

            var allAgents = FindObjectsByType<KartAgent>(FindObjectsSortMode.None);
            var trainingAgents = new List<KartAgent>();
            for (int i = 0; i < allAgents.Length; i++)
            {
                var agent = allAgents[i];
                if (agent == null || agent.Mode != AgentMode.Training)
                    continue;

                trainingAgents.Add(agent);
            }

            if (trainingAgents.Count <= 1)
                return Vector3.zero;

            trainingAgents.Sort(CompareAgentsForSpawnOrder);

            var laneIndex = trainingAgents.IndexOf(this);
            if (laneIndex < 0)
                return Vector3.zero;

            var columns = Mathf.Max(1, TrainingSpawnColumns);
            var columnIndex = laneIndex % columns;
            var rowIndex = laneIndex / columns;
            var centeredLane = columnIndex - (columns - 1) * 0.5f;
            var right = spawnRotation * Vector3.right;
            var backwards = -(spawnRotation * Vector3.forward);
            return right * (centeredLane * TrainingSpawnSpacing) + backwards * (rowIndex * TrainingSpawnRowSpacing);
        }

        static int CompareAgentsForSpawnOrder(KartAgent left, KartAgent right)
        {
            if (ReferenceEquals(left, right))
                return 0;

            if (left == null)
                return -1;

            if (right == null)
                return 1;

            return string.Compare(GetHierarchyPath(left.transform), GetHierarchyPath(right.transform), StringComparison.Ordinal);
        }

        static string GetHierarchyPath(Transform currentTransform)
        {
            if (currentTransform == null)
                return string.Empty;

            var hierarchySegments = new List<string>();
            while (currentTransform != null)
            {
                hierarchySegments.Add($"{currentTransform.GetSiblingIndex():D4}-{currentTransform.name}");
                currentTransform = currentTransform.parent;
            }

            hierarchySegments.Reverse();
            return string.Join("/", hierarchySegments);
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
