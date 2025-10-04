const posts = [
  { title: "about me", file: "me.md" },
  { title: "NV-Embed", file: "nvEmbed.md" },
];

const postList = document.getElementById("post-list");
const postContent = document.getElementById("post-content");
const backButton = document.getElementById("back-button");
const blogTitle = document.getElementById("blog-title");
// const mdParser = window.markdownit().use(window.markdownitMathjax3);
const mdParser = window.markdownit();

async function showPost(post) {
  try {
    const res = await fetch(`posts/${post.file}`);
    if (!res.ok) throw new Error("Failed to load post");

    const md = await res.text();
    postContent.innerHTML = mdParser.render(md);

    // Render Math
    if (window.MathJax) {
        MathJax.typesetPromise([postContent])
          .then(() => console.log("Math rendered!"))
          .catch((err) => console.error("MathJax error:", err));
    }

    postList.style.display = "none";
    blogTitle.textContent = post.title;
    postContent.style.display = "block";
    backButton.style.display = "inline-block";
  } catch (err) {
    postContent.innerHTML = `<p>Error loading post: ${err.message}</p>`;
  }
}

if (postList && postContent && backButton) {
  posts.forEach(post => {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = "#";
    a.textContent = post.title;

    a.addEventListener("click", e => {
      e.preventDefault();
      showPost(post);
    });

    li.appendChild(a);
    postList.appendChild(li);
  });

  backButton.addEventListener("click", () => {
    postContent.style.display = "none";
    postList.style.display = "block";
    blogTitle.textContent = "Blog";
    backButton.style.display = "none";
    postContent.innerHTML = "";
  });
}
