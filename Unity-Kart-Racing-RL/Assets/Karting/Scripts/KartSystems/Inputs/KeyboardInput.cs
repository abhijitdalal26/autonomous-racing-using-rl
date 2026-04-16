using UnityEngine;

namespace KartGame.KartSystems {

    public class KeyboardInput : BaseInput
    {
        public override InputData GenerateInput() {
            return new InputData
            {
                Accelerate = Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow),
                Brake = Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow),
                TurnInput = (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow)) ? -1f : 
                            (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow)) ? 1f : 0f
            };
        }
    }
}
