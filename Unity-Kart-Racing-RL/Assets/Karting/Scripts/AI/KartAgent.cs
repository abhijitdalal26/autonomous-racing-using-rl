using KartGame.KartSystems;
using System;
using System.Collections.Generic;
using System.IO;
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
        static string s_ClearedEpisodeTracePath;

#region Training Modes
        [Tooltip("Are we training the agent or is the agent production ready?")]
        public AgentMode Mode = AgentMode.Training;
        [Tooltip("What is the initial checkpoint the agent will go to? This value is only for inferencing.")]
        public ushort InitCheckpointIndex;
        [Tooltip("When enabled, training episodes can begin from any checkpoint. Disable to always start from checkpoint 0.")]
        public bool RandomizeTrainingStartCheckpoint;
        [Tooltip("Checkpoint index used for training when random starts are disabled. Set to 0 for the first checkpoint.")]
        public ushort TrainingStartCheckpointIndex;
        [Tooltip("When enabled, agents that were duplicated from prefabs can rebuild their checkpoint list from the scene.")]
        public bool AutoAssignSceneCheckpoints = true;
        [Tooltip("When enabled, only the primary training kart stays active and any duplicate training copies disable themselves.")]
        public bool SoloTrainingAgent = true;
        [Tooltip("How far apart training karts should be placed sideways when they share the same checkpoint reset.")]
        public float TrainingSpawnSpacing = 4f;
        [Tooltip("How far behind the first checkpoint each extra training row should be placed.")]
        public float TrainingSpawnRowSpacing = 6f;
        [Tooltip("Maximum number of training karts to place side by side before starting a new row.")]
        public int TrainingSpawnColumns = 3;
        [Tooltip("How many training action steps to ignore crash-ending ray hits after each reset.")]
        public int TrainingResetGraceSteps = 25;
        [Tooltip("When enabled, a training episode ends when the agent returns to the checkpoint where the episode started.")]
        public bool EndEpisodeOnLastCheckpoint = true;
        [Tooltip("When enabled, training agents that leave the drivable track are penalized and reset.")]
        public bool EndEpisodeWhenLeavingTrack = true;
        [Tooltip("Optional explicit training track config. If empty, the scene checkpoints are auto-discovered.")]
        public TrainingTrackConfig TrackConfig;
        [Tooltip("When enabled, the agent will start from its current position in the scene instead of teleporting to a spawn point or checkpoint.")]
        public bool UseScenePositionOnStart;

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
        [Tooltip("Penalty applied when a training agent leaves the drivable track.")]
        public float OffTrackPenalty = -1f;
        #endregion

        #region ResetParams
        [Header("Inference Reset Params")]
        [Tooltip("What is the unique mask that the agent should detect when it falls out of the track?")]
        public LayerMask OutOfBoundsMask;
        [Tooltip("What are the layers we want to detect for the track and the ground?")]
        public LayerMask TrackMask;
        [Tooltip("How far should the ray be when casted? For larger karts - this value should be larger too.")]
        public float GroundCastDistance;
        [Tooltip("Fallback downward ray distance used in training when GroundCastDistance is not configured.")]
        public float TrainingGroundCheckDistance = 5f;
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
        [Tooltip("When enabled, a structured episode trace is written to the project results folder for debugging.")]
        public bool WriteEpisodeTrace = true;
        [Tooltip("File name used for the structured episode trace inside the project's results folder.")]
        public string EpisodeTraceFileName = "episode_trace.log";
        [Tooltip("When enabled, the episode trace file is cleared once when Play mode starts.")]
        public bool ClearEpisodeTraceOnPlay = true;
#endregion

        ArcadeKart m_Kart;
        bool m_Acceleration;
        bool m_Brake;
        float m_Steering;
        int m_CheckpointIndex;
        int m_EpisodeStartCheckpointIndex;
        int m_CheckpointsPassedThisEpisode;

        bool m_EndEpisode;
        float m_LastAccumulatedReward;
        int m_StepsSinceReset;
        int m_EpisodeNumber;
        string m_PendingEndReason;
        string m_EpisodeTracePath;

        void Awake()
        {
            m_Kart = GetComponent<ArcadeKart>();
            if (AgentSensorTransform == null) AgentSensorTransform = transform;
            if (TrackConfig == null) TrackConfig = TrainingTrackConfig.Resolve();
            InitializeEpisodeTrace();

            if (Mode == AgentMode.Training && SoloTrainingAgent && ShouldDisableAsExtraTrainingAgent())
            {
                TraceEvent("agent-disabled", "Disabled because SoloTrainingAgent kept another training kart active.");
                gameObject.SetActive(false);
                return;
            }

            RefreshTrainingCheckpointsFromScene();
        }

        void InitializeEpisodeTrace()
        {
            if (!WriteEpisodeTrace)
                return;

            var resultsDirectory = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "results"));
            Directory.CreateDirectory(resultsDirectory);
            m_EpisodeTracePath = Path.Combine(resultsDirectory, EpisodeTraceFileName);

            if (ClearEpisodeTraceOnPlay && !string.Equals(s_ClearedEpisodeTracePath, m_EpisodeTracePath, StringComparison.OrdinalIgnoreCase))
            {
                File.WriteAllText(m_EpisodeTracePath, string.Empty);
                s_ClearedEpisodeTracePath = m_EpisodeTracePath;
            }

            TraceEvent("agent-awake", $"mode={Mode}");
        }

        void TraceEvent(string eventName, string details = "")
        {
            if (!WriteEpisodeTrace || string.IsNullOrEmpty(m_EpisodeTracePath))
                return;

            try
            {
                var position = transform != null ? transform.position : Vector3.zero;
                var traceLine =
                    $"{DateTime.Now:O}\tepisode={m_EpisodeNumber}\tstep={m_StepsSinceReset}\tevent={eventName}\tcheckpoint={m_CheckpointIndex}\tpos=({position.x:F2},{position.y:F2},{position.z:F2})";

                if (!string.IsNullOrWhiteSpace(details))
                    traceLine += "\t" + details.Replace('\t', ' ');

                File.AppendAllText(m_EpisodeTracePath, traceLine + Environment.NewLine);
            }
            catch (Exception traceException)
            {
                WriteEpisodeTrace = false;
                Debug.LogWarning($"Failed to write episode trace: {traceException.Message}");
            }
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
                m_EpisodeStartCheckpointIndex = m_CheckpointIndex;
                m_CheckpointsPassedThisEpisode = 0;
                if (TryGetCheckpointCollider(m_CheckpointIndex, out var checkpointCollider))
                {
                    // ResetToCheckpoint(checkpointCollider); // Commented out to allow starting from Editor position
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
                TraceEvent("episode-end", m_PendingEndReason);
                AddReward(m_LastAccumulatedReward);
                m_PendingEndReason = null;
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
                    if (IsOffTrack(out var hit))
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
                case AgentMode.Training:
                    if (!EndEpisodeWhenLeavingTrack || IsWithinTrainingResetGracePeriod())
                        break;

                    if (IsOffTrack(out _))
                    {
                        Debug.LogWarning($"{name} ended episode because it was detected off the drivable track.");
                        TraceEvent("episode-end", "reason=off-track");
                        AddReward(OffTrackPenalty);
                        EndEpisode();
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

            if (ShowRaycasts)
            {
                Debug.Log(
                    $"{name} touched checkpoint '{other.name}' (index {index}). " +
                    $"Current index: {m_CheckpointIndex}. Expected next: {expectedCheckpointIndex}.");
            }
            TraceEvent("checkpoint-touch",
                $"name={other.name};index={index};expected={expectedCheckpointIndex};current={m_CheckpointIndex}");

            if (index == m_CheckpointIndex) return; // Ignore if we hit the checkpoint we are already at!

            if (touchedExpectedCheckpoint)
            {
                AddReward(PassCheckpointReward);
                m_CheckpointIndex = index;
                m_CheckpointsPassedThisEpisode++;
                var returnedToEpisodeStartCheckpoint = index == m_EpisodeStartCheckpointIndex &&
                                                      m_CheckpointsPassedThisEpisode >= Colliders.Length;

                if (ShowRaycasts)
                {
                    Debug.Log($"{name} advanced to checkpoint {m_CheckpointIndex}.");
                }
                TraceEvent("checkpoint-advance", $"name={other.name};new_index={m_CheckpointIndex}");

                if (Mode == AgentMode.Training && EndEpisodeOnLastCheckpoint && returnedToEpisodeStartCheckpoint)
                {
                    TraceEvent("episode-end", "reason=completed-lap");
                    EndEpisode();
                }

                return;
            }

            if (Mode == AgentMode.Training)
            {
                Debug.LogWarning(
                    $"{name} hit checkpoint '{other.name}' out of order. " +
                    $"Expected '{Colliders[expectedCheckpointIndex].name}', but touched '{Colliders[index].name}'.");
                TraceEvent("checkpoint-out-of-order",
                    $"touched={other.name};expected={Colliders[expectedCheckpointIndex].name};index={index}");
                AddReward(WrongCheckpointPenalty);
                if (EndEpisodeOnWrongCheckpoint)
                {
                    TraceEvent("episode-end", $"reason=wrong-checkpoint;touched={other.name}");
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
                if (IsSameCheckpoint(Colliders[i], checkPoint))
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
                            m_PendingEndReason =
                                $"reason=sensor-hit;sensor={i};collider={hitInfo.collider.name};distance={hitInfo.distance:F2};threshold={current.HitValidationDistance:F2}";
                            if (ShowRaycasts)
                            {
                                Debug.LogWarning(
                                    $"{name} ended episode because sensor {i} detected '{hitInfo.collider.name}' " +
                                    $"at distance {hitInfo.distance:F2}, inside threshold {current.HitValidationDistance:F2}.");
                            }
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
                    m_EpisodeStartCheckpointIndex = m_CheckpointIndex;
                    m_CheckpointsPassedThisEpisode = 0;
                    m_EpisodeNumber++;
                    m_PendingEndReason = null;
                    if (UseScenePositionOnStart && m_StepsSinceReset == 0)
                    {
                        TraceEvent("episode-begin", "spawn=ScenePosition;UseScenePositionOnStart=true");
                        ClearMotionAndInput();
                        break;
                    }
                    var collider = Colliders[m_CheckpointIndex];
                    if (TryGetEpisodeSpawnPoint(out var spawnPoint) && !RandomizeTrainingStartCheckpoint)
                    {
                        ResetToSpawnPoint(spawnPoint);
                        if (collider != null && collider.bounds.Contains(transform.position))
                        {
                            Debug.LogWarning(
                                $"{name} spawn point appears to be inside or too close to checkpoint '{collider.name}'. " +
                                "Move EpisodeSpawnPoint farther back so the kart crosses the trigger from outside.");
                        }
                        TraceEvent("episode-begin",
                            $"spawn=EpisodeSpawnPoint;start_checkpoint_index={m_CheckpointIndex};target={Colliders[(m_CheckpointIndex + 1) % Colliders.Length].name}");
                    }
                    else
                    {
                        ResetToCheckpoint(collider);
                        TraceEvent("episode-begin",
                            $"spawn=CheckpointReset;start_checkpoint_index={m_CheckpointIndex};target={Colliders[(m_CheckpointIndex + 1) % Colliders.Length].name}");
                    }
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

        void ResetToSpawnPoint(Transform spawnPoint)
        {
            if (spawnPoint == null)
                return;

            var spawnRotation = spawnPoint.rotation;
            var spawnPosition = spawnPoint.position + GetTrainingSpawnOffset(spawnRotation);
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

            if (HasConfiguredCheckpointList())
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
            if (HasConfiguredCheckpointList())
                return true;

            if (!(AutoAssignSceneCheckpoints || Mode == AgentMode.Training))
                return false;

            var activeTrackConfig = ResolveTrackConfig();
            if (activeTrackConfig == null || !activeTrackConfig.TryGetOrderedCheckpoints(out var discoveredCheckpoints))
                return false;

            Colliders = discoveredCheckpoints;
            m_CheckpointIndex = ClampCheckpointIndex(m_CheckpointIndex);
            return true;
        }

        bool HasConfiguredCheckpointList()
        {
            if (Colliders == null || Colliders.Length == 0)
                return false;

            for (int i = 0; i < Colliders.Length; i++)
            {
                if (Colliders[i] != null)
                    return true;
            }

            return false;
        }

        TrainingTrackConfig ResolveTrackConfig()
        {
            if (TrackConfig == null)
                TrackConfig = TrainingTrackConfig.Resolve();

            return TrackConfig;
        }

        bool TryGetEpisodeSpawnPoint(out Transform spawnPoint)
        {
            spawnPoint = null;
            var activeTrackConfig = ResolveTrackConfig();
            if (activeTrackConfig == null)
                return false;

            return activeTrackConfig.TryGetEpisodeSpawnPoint(out spawnPoint);
        }

        bool IsOffTrack(out RaycastHit hit)
        {
            var groundCheckDistance = GroundCastDistance > 0f ? GroundCastDistance : TrainingGroundCheckDistance;
            var trackMask = TrackMask.value != 0 ? TrackMask : (LayerMask)k_DefaultRespawnMask;
            var rayOrigin = transform.position + Vector3.up;

            if (ShowRaycasts)
                Debug.DrawRay(rayOrigin, Vector3.down * groundCheckDistance, Color.cyan);

            if (!Physics.Raycast(rayOrigin, Vector3.down, out hit, groundCheckDistance, trackMask, QueryTriggerInteraction.Ignore))
                return true;

            if (OutOfBoundsMask.value != 0 && ((1 << hit.collider.gameObject.layer) & OutOfBoundsMask) > 0)
                return true;

            return false;
        }

        int GetTrainingStartCheckpointIndex()
        {
            if (Colliders == null || Colliders.Length == 0)
                return 0;

            if (RandomizeTrainingStartCheckpoint)
                return Random.Range(0, Colliders.Length);

            if (TryGetEpisodeSpawnPoint(out _))
            {
                // With a dedicated spawn point before the line, keep the finish/start line as the
                // last ordered checkpoint so the first expected checkpoint is index 0.
                return Colliders.Length - 1;
            }

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

            // Only apply spawn spacing if we are at the very first checkpoint.
            // When resetting at corners, the grid formation pushes high-index karts backwards off the track mesh.
            if (m_CheckpointIndex != 0 || RandomizeTrainingStartCheckpoint)
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

        static bool IsSameCheckpoint(Collider expected, Collider actual)
        {
            if (expected == null || actual == null)
                return false;

            if (expected == actual)
                return true;

            if (expected.transform == actual.transform)
                return true;

            if (actual.transform.IsChildOf(expected.transform))
                return true;

            return expected.transform.IsChildOf(actual.transform);
        }

        bool ShouldDisableAsExtraTrainingAgent()
        {
            var allAgents = FindObjectsByType<KartAgent>(FindObjectsSortMode.None);
            KartAgent preferredAgent = null;

            for (int i = 0; i < allAgents.Length; i++)
            {
                var agent = allAgents[i];
                if (agent == null || agent.Mode != AgentMode.Training || !agent.gameObject.activeInHierarchy)
                    continue;

                if (preferredAgent == null || IsPreferredTrainingAgent(agent, preferredAgent))
                {
                    preferredAgent = agent;
                }
            }

            return preferredAgent != null && preferredAgent != this;
        }

        static bool IsPreferredTrainingAgent(KartAgent candidate, KartAgent currentBest)
        {
            if (candidate == null)
                return false;

            if (currentBest == null)
                return true;

            var candidateName = candidate.gameObject.name;
            var currentBestName = currentBest.gameObject.name;

            var candidateIsExactKartAi = string.Equals(candidateName, "Kart_AI", StringComparison.OrdinalIgnoreCase);
            var currentBestIsExactKartAi = string.Equals(currentBestName, "Kart_AI", StringComparison.OrdinalIgnoreCase);
            if (candidateIsExactKartAi != currentBestIsExactKartAi)
                return candidateIsExactKartAi;

            var candidatePath = GetHierarchyPath(candidate.transform);
            var currentBestPath = GetHierarchyPath(currentBest.transform);
            return string.Compare(candidatePath, currentBestPath, StringComparison.Ordinal) < 0;
        }
    }
}
