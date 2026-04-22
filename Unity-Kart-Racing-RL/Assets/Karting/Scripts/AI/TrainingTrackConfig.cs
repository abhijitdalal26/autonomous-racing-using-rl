using System;
using System.Collections.Generic;
using UnityEngine;

namespace KartGame.AI
{
    /// <summary>
    /// Centralizes training track discovery so agents can use a stable ordered checkpoint list.
    /// </summary>
    public class TrainingTrackConfig : MonoBehaviour
    {
        const string k_DefaultCheckpointName = "Checkpoint";

        static TrainingTrackConfig s_RuntimeInstance;

        [Tooltip("Ordered checkpoint colliders used for lap progression. If empty, the scene checkpoints are auto-discovered.")]
        public Collider[] OrderedCheckpoints;
        [Tooltip("Maximum number of ordered checkpoints to use. Extra discovered checkpoints are ignored.")]
        public int MaxOrderedCheckpoints = 3;
        [Tooltip("Whether runtime auto-discovery should fill missing checkpoint references.")]
        public bool AutoDiscoverSceneReferences = true;
        [Tooltip("Optional training spawn point placed before the first checkpoint. When assigned, keep the finish/start line as the last ordered checkpoint.")]
        public Transform EpisodeSpawnPoint;

        public static TrainingTrackConfig Resolve()
        {
            var existingConfig = FindFirstObjectByType<TrainingTrackConfig>();
            if (existingConfig != null)
            {
                s_RuntimeInstance = existingConfig;
                return existingConfig;
            }

            if (s_RuntimeInstance != null)
                return s_RuntimeInstance;

            var runtimeConfigObject = new GameObject("TrainingTrackConfig (Runtime)");
            runtimeConfigObject.hideFlags = HideFlags.DontSave;
            s_RuntimeInstance = runtimeConfigObject.AddComponent<TrainingTrackConfig>();
            return s_RuntimeInstance;
        }

        public bool TryGetOrderedCheckpoints(out Collider[] checkpoints)
        {
            checkpoints = null;

            var configuredCheckpoints = LimitCheckpointCount(FilterValidColliders(OrderedCheckpoints));
            if (configuredCheckpoints.Count == 0 && AutoDiscoverSceneReferences)
            {
                configuredCheckpoints = LimitCheckpointCount(DiscoverSceneCheckpoints());
                OrderedCheckpoints = configuredCheckpoints.ToArray();
            }

            if (configuredCheckpoints.Count == 0)
                return false;

            checkpoints = configuredCheckpoints.ToArray();
            return true;
        }

        public bool TryGetEpisodeSpawnPoint(out Transform spawnPoint)
        {
            spawnPoint = EpisodeSpawnPoint;
            return spawnPoint != null;
        }

        static List<Collider> FilterValidColliders(Collider[] source)
        {
            var validColliders = new List<Collider>();
            if (source == null)
                return validColliders;

            for (int i = 0; i < source.Length; i++)
            {
                if (source[i] != null)
                    validColliders.Add(source[i]);
            }

            return validColliders;
        }

        List<Collider> LimitCheckpointCount(List<Collider> checkpoints)
        {
            if (checkpoints == null)
                return new List<Collider>();

            var maxCheckpoints = Mathf.Max(1, MaxOrderedCheckpoints);
            if (checkpoints.Count <= maxCheckpoints)
                return checkpoints;

            return checkpoints.GetRange(0, maxCheckpoints);
        }

        static List<Collider> DiscoverSceneCheckpoints()
        {
            var discoveredCheckpoints = new List<Collider>();
            var sceneColliders = FindObjectsByType<Collider>(FindObjectsSortMode.None);
            for (int i = 0; i < sceneColliders.Length; i++)
            {
                var collider = sceneColliders[i];
                if (collider == null || !collider.gameObject.activeInHierarchy)
                    continue;

                if (collider.transform.name.IndexOf(k_DefaultCheckpointName, StringComparison.OrdinalIgnoreCase) < 0)
                    continue;

                discoveredCheckpoints.Add(collider);
            }

            discoveredCheckpoints.Sort(CompareCheckpointColliders);
            return discoveredCheckpoints;
        }

        static int CompareCheckpointColliders(Collider left, Collider right)
        {
            if (ReferenceEquals(left, right))
                return 0;

            if (left == null)
                return -1;

            if (right == null)
                return 1;

            ParseCheckpointName(left.transform.name, out var leftBaseName, out var leftIndex);
            ParseCheckpointName(right.transform.name, out var rightBaseName, out var rightIndex);

            var baseNameCompare = string.Compare(leftBaseName, rightBaseName, StringComparison.OrdinalIgnoreCase);
            if (baseNameCompare != 0)
                return baseNameCompare;

            var indexCompare = leftIndex.CompareTo(rightIndex);
            if (indexCompare != 0)
                return indexCompare;

            return string.Compare(GetHierarchyPath(left.transform), GetHierarchyPath(right.transform), StringComparison.Ordinal);
        }

        static void ParseCheckpointName(string checkpointName, out string baseName, out int index)
        {
            var trimmedName = string.IsNullOrWhiteSpace(checkpointName) ? string.Empty : checkpointName.Trim();
            baseName = trimmedName;
            index = 0;

            var openParen = trimmedName.LastIndexOf('(');
            var closeParen = trimmedName.LastIndexOf(')');
            if (openParen >= 0 && closeParen == trimmedName.Length - 1 && openParen < closeParen)
            {
                var numericSuffix = trimmedName.Substring(openParen + 1, closeParen - openParen - 1);
                if (int.TryParse(numericSuffix, out index))
                {
                    baseName = trimmedName.Substring(0, openParen).TrimEnd();
                    return;
                }
            }

            var suffixStart = trimmedName.Length - 1;
            while (suffixStart >= 0 && char.IsDigit(trimmedName[suffixStart]))
                suffixStart--;

            if (suffixStart < trimmedName.Length - 1 && int.TryParse(trimmedName.Substring(suffixStart + 1), out index))
            {
                baseName = trimmedName.Substring(0, suffixStart + 1).TrimEnd(' ', '-', '_');
            }
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
    }
}
