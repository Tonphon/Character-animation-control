#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/shader_m.h>
#include <learnopengl/camera.h>
#include <learnopengl/animator.h>
#include <learnopengl/model_animation.h>

#include <iostream>
#include <string>
#include <vector>

// ------------- settings -------------
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

// ------------- callbacks ------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height);

// ------------- timing ---------------
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// ------------- input ----------------
struct Input {
    float axisRight = 0.0f;
    float axisForward = 0.0f;
    bool  jumpPressed = false;
} input;

// ------------- player ---------------
enum class AnimState { Idle, Running, Jumping };

struct Player {
    glm::vec3 pos{ 0.0f, 0.0f, 0.0f };
    glm::vec3 vel{ 0.0f, 0.0f, 0.0f };
    float yaw = 0.0f;
    float moveSpeed = 3.5f;
    float jumpSpeed = 5.0f;
    float gravity = 12.0f;
    bool  onGround = true;
    AnimState state = AnimState::Idle;

    void updatePhysics(float dt) {
        glm::vec3 wish = glm::vec3(input.axisRight, 0.0f, -input.axisForward);
        glm::vec3 horizVel(0.0f);

        if (glm::length2(wish) > 1e-6f) {
            wish = glm::normalize(wish);
            horizVel = wish * moveSpeed;
            yaw = glm::degrees(atan2(wish.x, -wish.z));
            std::cout << yaw;
            if (yaw == 0.0f || yaw == 180.0f) yaw = yaw + 180.0f;
            else if ( yaw == 45.0f || yaw == -135.0f) yaw = yaw + 90.0f;
            else if (yaw == -45.0f || yaw == 135.0f) yaw = yaw - 90.0f;
            if (onGround) state = AnimState::Running;
        }
        else if (onGround) {
            state = AnimState::Idle;
        }

        if (input.jumpPressed && onGround) {
            vel.y = jumpSpeed;
            onGround = false;
            state = AnimState::Jumping;
        }

        if (!onGround) vel.y -= gravity * dt;

        pos += horizVel * dt;
        pos.y += vel.y * dt;

        // ground at y=0
        if (pos.y <= 0.0f) {
            pos.y = 0.0f;
            vel.y = 0.0f;
            if (!onGround) {
                onGround = true;
                state = (glm::length2(wish) > 1e-6f) ? AnimState::Running : AnimState::Idle;
            }
        }
    }
} player;

// ------------- fixed camera ----------
glm::mat4 computeFixedChaseCamView()
{
    const glm::vec3 camOffset(0.0f, 3.0f, 6.5f);
    const glm::vec3 lookAhead(0.0f, 1.2f, -4.0f);
    glm::vec3 camPos = player.pos + camOffset;
    glm::vec3 target = player.pos + lookAhead;
    return glm::lookAt(camPos, target, glm::vec3(0.0f, 1.0f, 0.0f));
}

// ------------- utils ------------------
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
unsigned int LoadTexture2D(const std::string& fullPath, bool flip = true)
{
    if (flip) stbi_set_flip_vertically_on_load(true);
    int w, h, n;
    unsigned char* data = stbi_load(fullPath.c_str(), &w, &h, &n, 0);
    if (!data) {
        std::cerr << "Failed to load texture: " << fullPath
            << " reason: " << stbi_failure_reason() << std::endl;
        return 0;
    }
    GLenum format = (n == 1) ? GL_RED : (n == 3) ? GL_RGB : GL_RGBA;
    unsigned int tex; glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    stbi_image_free(data);
    return tex;
}

// Plane VAO with dummy skinning attributes so we can reuse anim shader
struct Plane {
    unsigned int VAO = 0, VBO = 0, EBO = 0;
    unsigned int texture = 0;

    void init(const std::string& texPath)
    {
        // 2 triangles (indexed), with texcoords tiling
        const float SIZE = 200.0f; // 200x200 world units
        const float T = 40.0f;     // texture tiles across plane
        // pos (3), normal (3), tex (2), boneIDs (4 ints packed as floats), weights (4)
        struct V { float px, py, pz, nx, ny, nz, u, v, bid0, bid1, bid2, bid3, w0, w1, w2, w3; };
        const V verts[] = {
            {-SIZE, 0,  SIZE, 0,1,0, 0, T, 0,0,0,0, 1,0,0,0},
            { SIZE, 0,  SIZE, 0,1,0, T, T, 0,0,0,0, 1,0,0,0},
            { SIZE, 0, -SIZE, 0,1,0, T, 0, 0,0,0,0, 1,0,0,0},
            {-SIZE, 0, -SIZE, 0,1,0, 0, 0, 0,0,0,0, 1,0,0,0},
        };
        const unsigned int idx[] = { 0,1,2, 0,2,3 };

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

        GLsizei stride = sizeof(V);
        // layout matches anim vertex shader:
        // 0: position
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(V, px));
        // 1: normal
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(V, nx));
        // 2: texcoords
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(V, u));
        // 3: bone IDs (ivec4)
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(V, bid0));
        // 4: weights (vec4)
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(V, w0));

        glBindVertexArray(0);

        texture = LoadTexture2D(texPath, true);
    }

    void draw(Shader& animShader)
    {

        glm::mat4 I(1.0f);
        for (int i = 0; i < 100; ++i)
            animShader.setMat4("finalBonesMatrices[" + std::to_string(i) + "]", I);

        glm::mat4 model(1.0f);
        animShader.setMat4("model", model);

        animShader.setInt("texture_diffuse1", 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
} gPlane;

// ------------- main -------------------
int main()
{
    // GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Playable Character (Fixed Cam + Floor)", nullptr, nullptr);
    if (!window) { std::cerr << "Failed to create window\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD\n"; return -1;
    }
    glEnable(GL_DEPTH_TEST);

    Shader animShader("anim_model.vs", "anim_model.fs");

    gPlane.init("C:/Users/User/source/repos/LearnOpenGL/resources/textures/marble.jpg");

    const std::string base = "C:/Users/User/source/repos/LearnOpenGL/resources/objects/player/";
    Model playerModel(base + "Idle.dae");
    Animation idleAnim(base + "Idle.dae", &playerModel);
    Animation runAnim(base + "Running.dae", &playerModel);
    Animation jumpAnim(base + "Jump.dae", &playerModel);
    Animator animator(&idleAnim);

    player.yaw = 180.0f;

    // loop
    bool prevSpace = false;
    while (!glfwWindowShouldClose(window))
    {
        float now = (float)glfwGetTime();
        deltaTime = now - lastFrame;
        lastFrame = now;

        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

        float f = 0.0f, r = 0.0f;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) f += 1.0f;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) f -= 1.0f;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) r += 1.0f;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) r -= 1.0f;
        glm::vec2 ax(r, f);
        if (glm::length(ax) > 1e-6f) ax = glm::normalize(ax); else ax = glm::vec2(0.0f);
        input.axisRight = ax.x;
        input.axisForward = ax.y;

        bool spaceDown = (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);
        input.jumpPressed = spaceDown && !prevSpace;
        prevSpace = spaceDown;

        player.updatePhysics(deltaTime);

        static AnimState lastState = AnimState::Idle;
        if (player.state != lastState) {
            switch (player.state) {
            case AnimState::Idle:    animator.PlayAnimation(&idleAnim);  break;
            case AnimState::Running: animator.PlayAnimation(&runAnim);   break;
            case AnimState::Jumping: animator.PlayAnimation(&jumpAnim);  break;
            }
            lastState = player.state;
        }
        animator.UpdateAnimation(deltaTime);

        // render
        glClearColor(0.06f, 0.06f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        animShader.use();
        glm::mat4 projection = glm::perspective(glm::radians(50.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 400.0f);
        glm::mat4 view = computeFixedChaseCamView();
        animShader.setMat4("projection", projection);
        animShader.setMat4("view", view);

        // --- draw floor ---
        gPlane.draw(animShader);

        // --- set animated bones & draw player ---
        auto transforms = animator.GetFinalBoneMatrices();
        for (int i = 0; i < (int)transforms.size(); ++i)
            animShader.setMat4("finalBonesMatrices[" + std::to_string(i) + "]", transforms[i]);

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, player.pos);
        model = glm::rotate(model, glm::radians(player.yaw), glm::vec3(0, 1, 0));
        model = glm::scale(model, glm::vec3(0.6f));
        animShader.setMat4("model", model);

        // Ensure same sampler for the character materials
        animShader.setInt("texture_diffuse1", 0);

        playerModel.Draw(animShader);

        glfwSwapBuffers(window);
    }

    glfwTerminate();
    return 0;
}

void framebuffer_size_callback(GLFWwindow* /*window*/, int width, int height)
{
    glViewport(0, 0, width, height);
}
